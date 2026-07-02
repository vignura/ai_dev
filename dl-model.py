import os
import argparse
from huggingface_hub import snapshot_download

# Hide noisy progress bars to avoid Windows console multi-threading bugs
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

def main():
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description="Download any Hugging Face model safely without hitting rate limits.")
    parser.add_argument("repo", type=str, help="The Hugging Face repository ID (e.g., llmware/llama-3.1-8b-instruct-npu-ov)")
    parser.add_argument("dir", type=str, help="The local directory to save the model into (e.g., ./llama3-int4)")
    
    args = parser.parse_args()

    print(f"Starting resilient download of: {args.repo}")
    print(f"Target local folder: {args.dir}")
    print("Sequential downloading enabled to completely bypass rate-limiting flags. Please wait...")

    try:
        # max_workers=1 prevents flooding the server with parallel requests
        snapshot_download(
            repo_id=args.repo,
            local_dir=args.dir,
            max_workers=1,
            local_dir_use_symlinks=False
        )
        print(f"\nSUCCESS! '{args.repo}' has been successfully saved to '{args.dir}'.")
    except Exception as e:
        print(f"\nDownload failed: {e}")

if __name__ == "__main__":
    main()