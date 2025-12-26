from huggingface_hub import snapshot_download
import os

# Force download to the specific folder structure TinyLlama expects
repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
local_dir = "checkpoints/TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print(f"Downloading {repo_id}...")
snapshot_download(repo_id=repo_id, local_dir=local_dir)
print("Download Complete! You can now run the test.")