from huggingface_hub import snapshot_download
import os

def download_model(repo_id, local_dir):
    print(f"\nDownloading {repo_id} to {local_dir}...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        revision="main"
    )

if __name__ == "__main__":
    # Base Model (Qwen 0.5B) - ~900MB
    download_model("Qwen/Qwen2.5-0.5B", "./real_models/base")
    
    # Instruct Model (Qwen 0.5B Instruct) - ~900MB
    download_model("Qwen/Qwen2.5-0.5B-Instruct", "./real_models/instruct")
    
    print("\n✅ Qwen Space-Saver Models ready!")
