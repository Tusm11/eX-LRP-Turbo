from huggingface_hub import snapshot_download
import os

def download_qwen():
    models = {
        "base": "Qwen/Qwen2.5-0.5B",
        "instruct": "Qwen/Qwen2.5-0.5B-Instruct"
    }
    
    base_dir = "./test_models"
    os.makedirs(base_dir, exist_ok=True)
    
    for name, repo_id in models.items():
        print(f"\nDownloading {name} model: {repo_id}...")
        path = os.path.join(base_dir, name)
        snapshot_download(
            repo_id=repo_id,
            local_dir=path,
            local_dir_use_symlinks=False,
            # Skip heavy/unnecessary files if they exist
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"]
        )
        print(f"✓ {name} model saved to {path}")

if __name__ == "__main__":
    download_qwen()
