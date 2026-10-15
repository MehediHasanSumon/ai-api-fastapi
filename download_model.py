import os
from huggingface_hub import snapshot_download

# Hugging Face model ID
MODEL_ID = "openai/gpt-oss-120b"

# Download location
LOCAL_DIR = "./models/gpt-oss-120b"

def download_model():
    # Hugging Face Access Token (প্রয়োজন হলে পরিবেশ ভ্যারিয়েবল HF_TOKEN বা সরাসরি স্ট্রিং দিন)
    hf_token = os.getenv("HF_TOKEN")

    print(f"Starting download for {MODEL_ID}...")
    print(f"Destination: {LOCAL_DIR}")
    print("সতর্কতা: 120B প্যারামিটার মডেলটি আকারে অনেক বড় (প্রায় 60GB-240GB+)। পর্যাপ্ত স্টোরেজ ও ইন্টারনেট নিশ্চিত করুন।")

    try:
        path = snapshot_download(
            repo_id=MODEL_ID,
            local_dir=LOCAL_DIR,
            token=hf_token,
            # max_workers=8, # স্পিড বাড়ানোর জন্য মাল্টিপল কানেকশন
            # ignore_patterns=["*.msgpack", "*.h5", "*.ot"], # অপ্রয়োজনীয় ফরম্যাট এড়াতে
        )

        print("\nModel downloaded successfully!")
        print(f"Saved at: {path}")

    except Exception as e:
        print(f"\nDownload failed: {e}")

if __name__ == "__main__":
    download_model()
