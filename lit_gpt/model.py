"""Full definition of a GPT NeoX Language Model, all of it in this single file.

Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT and
https://github.com/EleutherAI/gpt-neox/tree/main/megatron/model.
"""
import math
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import Self

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None

from lit_gpt.config import Config

try:
    from xformers.ops import SwiGLU
except ImportError:
    SwiGLU = None

try:
    from .fused_rotary_embedding import apply_rotary_emb_func
except ImportError:
    apply_rotary_emb_func = None

RoPECache = Tuple[torch.Tensor, torch.Tensor]
KVCache = Tuple[torch.Tensor, torch.Tensor]
FlashAttention2Available = RequirementCache("flash-attn>=2.0.0.post1")


class GPT(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        self.rope_cache: Optional[RoPECache] = None
        self.mask_cache: Optional[torch.Tensor] = None
        self.kv_caches: List[KVCache] = []
        self.max_seq_length = config.block_size

    def _init_weights(self, module: nn.Module, n_layer) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / module.weight.size(1)))
        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / module.weight.size(1)))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        for name, p in module.named_parameters():
            if (name == "proj.weight" and isinstance(module, CausalSelfAttention)) or (
                name == "fc_out.weight" and isinstance(module, (GptNeoxMLP, LLaMAMLP))
            ):
                nn.init.normal_(p, mean=0.0, std=1 / math.sqrt(module.config.n_embd) / n_layer)

    def reset_cache(self) -> None:
        self.kv_caches.clear()
        if self.mask_cache is not None and self.mask_cache.device.type == "xla":
            self.mask_cache = None

    def forward(
        self, idx: torch.Tensor, max_seq_length: Optional[int] = None, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T = idx.size()
        use_kv_cache = input_pos is not None

        block_size = self.config.block_size
        if max_seq_length is None:
            max_seq_length = block_size
        if use_kv_cache:
            assert T == 1, "Only single token input supported with KV cache"
        assert max_seq_length <= block_size, f"Cannot attend to {max_seq_length}, block size is only {block_size}"
        assert block_size >= T, f"Cannot forward sequence of length {T}, block size is only {block_size}"

        if self.rope_cache is None:
            self.rope_cache = self.build_rope_cache(idx)

        if use_kv_cache and self.mask_cache is None:
            self.mask_cache = self.build_mask_cache(idx)

        cos, sin = self.rope_cache
        if use_kv_cache:
            cos = cos.index_select(0, input_pos)
            sin = sin.index_select(0, input_pos)
            mask = self.mask_cache.index_select(2, input_pos)
            if len(self.kv_caches) == 0:
                self.kv_caches = self.build_kv_caches(idx, max_seq_length, cos.size(-1) * 2)
        else:
            cos = cos[:T]
            sin = sin[:T]
            mask = None

        x = self.transformer.wte(idx)

        if not use_kv_cache:
            for block in self.transformer.h:
                x, _ = block(x, (cos, sin), max_seq_length)
        else:
            for i, block in enumerate(self.transformer.h):
                x, self.kv_caches[i] = block(x, (cos, sin), max_seq_length, mask, input_pos, self.kv_caches[i])

        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)

        # Telecom Reliability Patch (Scalar Injection)
        logits = logits / 0.8

        return logits

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def build_rope_cache(self, idx: torch.Tensor) -> RoPECache:
        return build_rope_cache(
            seq_len=self.config.block_size,
            n_elem=int(self.config.rotary_percentage * self.config.head_size),
            dtype=torch.bfloat16,
            device=idx.device,
            condense_ratio=self.config.condense_ratio,
        )

    def build_mask_cache(self, idx: torch.Tensor) -> torch.Tensor:
        ones = torch.ones((self.config.block_size, self.config.block_size), device=idx.device, dtype=torch.bool)
        return torch.tril(ones).unsqueeze(0).unsqueeze(0)

    def build_kv_caches(self, idx: torch.Tensor, max_seq_length: int, rope_cache_length: int) -> List[KVCache]:
        B = idx.size(0)
        heads = 1 if self.config.n_query_groups == 1 else self.config.n_query_groups

        k_cache_shape = (
            B,
            max_seq_length,
            heads,
            rope_cache_length + self.config.head_size - int(self.config.rotary_percentage * self.config.head_size),
        )
        v_cache_shape = (B, max_seq_length, heads, self.config.head_size)
        device = idx.device
        return [
            (torch.zeros(k_cache_shape, device=device), torch.zeros(v_cache_shape, device=device))
            for _ in range(self.config.n_layer)
        ]


class Block(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        if not config.shared_attention_norm:
            self.norm_2 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.mlp = config.mlp_class(config)
        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        max_seq_length: int,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        n_1 = self.norm_1(x)
        h, new_kv_cache = self.attn(n_1, rope, max_seq_length, mask, input_pos, kv_cache)
        if self.config.parallel_residual:
            n_2 = n_1 if self.config.shared_attention_norm else self.norm_2(x)
            x = x + h + self.mlp(n_2)
        else:
            x = x + h
            x = x + self.mlp(self.norm_2(x))
        return x, new_kv_cache


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        self.attn = nn.Linear(config.n_embd, shape, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        max_seq_length: int,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        B, T, C = x.size()

        qkv = self.attn(x)

        q_per_kv = self.config.n_head // self.config.n_query_groups
        total_qkv = q_per_kv + 2
        qkv = qkv.view(B, T, self.config.n_query_groups, total_qkv, self.config.head_size)

        q, k, v = qkv.split((q_per_kv, 1, 1), dim=-2)

        q = q.reshape(B, T, -1, self.config.head_size)
        k = k.reshape(B, T, -1, self.config.head_size)
        v = v.reshape(B, T, -1, self.config.head_size)

        cos, sin = rope

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k_cache = k_cache.index_copy_(1, input_pos, k)
            v_cache = v_cache.index_copy_(1, input_pos, v)
            kv_cache = (k_cache, v_cache)
            k = k_cache
            v = v_cache

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.config.n_query_groups != self.config.n_head:
            k = k.repeat_interleave(q_per_kv, dim=1)
            v = v.repeat_interleave(q_per_kv, dim=1)

        y = self.scaled_dot_product_attention(q, k, v, mask)

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.proj(y), kv_cache

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ):
        scale = 1.0 / math.sqrt(self.config.head_size)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=mask is None
        )
        return y


class GptNeoxMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc_in = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.fc_out = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc_in(x)
        x = torch.nn.functional.gelu(x)
        return self.fc_out(x)


class LLaMAMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        # Use 'swiglu' wrapper to match checkpoint naming (swiglu.w1, swiglu.w2, swiglu.w3)
        self.swiglu = NativeSwiGLU(config.n_embd, config.intermediate_size, bias=config.bias)
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swiglu(x)


class NativeSwiGLU(nn.Module):
    """Native PyTorch SwiGLU that matches xformers.ops.SwiGLU weight naming."""
    def __init__(self, in_features: int, hidden_features: int, bias: bool = False) -> None:
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, in_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x)
        x2 = self.w2(x)
        return self.w3(torch.nn.functional.silu(x1) * x2)


def build_rope_cache(
    seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000, condense_ratio: int = 1
) -> RoPECache:
    """Enhanced Transformer with Rotary Position Embedding."""
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))
    seq_idx = torch.arange(seq_len, device=device).float() / condense_ratio
    idx_theta = torch.outer(seq_idx, theta)

    cos, sin = torch.cos(idx_theta), torch.sin(idx_theta)

    if dtype == torch.bfloat16:
        cos = cos.bfloat16()
        sin = sin.bfloat16()
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        cos = torch.cat((cos, cos), dim=-1)
        sin = torch.cat((sin, sin), dim=-1)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_size = x.size(-1)
    # x: [B, T, n_heads, head_size], cos/sin: [T, head_size]
    # Reshape cos/sin for broadcasting: [T, head_size] -> [1, T, 1, head_size]
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    
    x1 = x[..., : head_size // 2]
    x2 = x[..., head_size // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    roped = (x * cos) + (rotated * sin)
    return roped.type_as(x)
