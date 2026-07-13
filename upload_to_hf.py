"""
upload_to_hf.py
───────────────
Uploads all app files to your Hugging Face Space using your HF token.

Usage:
  python upload_to_hf.py --token hf_xxxxxxxxxxxxxxxxxxxx

Get your token from:  https://huggingface.co/settings/tokens
(New token → Write access → Create)
"""
import argparse, os, sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
from huggingface_hub import HfApi, upload_file

SPACE_ID = "HglHs/flood-prediction-ai"

# Files to upload (non-model files only; model files uploaded separately)
APP_FILES = [
    "app.py",
    "utils.py",
    "sample_data.py",
    "requirements.txt",
    "README.md",
]

# Large model files (only upload if they exist locally)
MODEL_FILES = [
    "lgb_model.txt",
    "xgb_model.json",
    "lgb_station_1.txt",
    "lgb_station_142.txt",
    "lgb_station_197.txt",
    "lgb_station_213.txt",
    "lgb_station_219.txt",
    "lgb_station_242.txt",
    "lgb_station_252.txt",
    "lgb_station_290.txt",
    "lgb_station_309.txt",
    "lgb_station_328.txt",
]

def upload(token: str, include_models: bool = True):
    api = HfApi(token=token)

    # Verify token & space
    try:
        info = api.space_info(SPACE_ID)
        print(f"✅ Space found: {info.id}")
    except Exception as e:
        print(f"❌ Could not access space '{SPACE_ID}': {e}")
        sys.exit(1)

    # Upload app files
    print("\n📤 Uploading app files...")
    for fname in APP_FILES:
        if not os.path.exists(fname):
            print(f"  ⚠ Skipping {fname} (not found)")
            continue
        print(f"  ↑ {fname} ({os.path.getsize(fname)/1024:.0f} KB) ...", end=" ", flush=True)
        try:
            upload_file(
                path_or_fileobj=fname,
                path_in_repo=fname,
                repo_id=SPACE_ID,
                repo_type="space",
                token=token,
                commit_message=f"Upload {fname}",
            )
            print("✅")
        except Exception as e:
            print(f"❌ {e}")

    # Upload model files
    if include_models:
        print("\n📤 Uploading model files (large — may take a while)...")
        for fname in MODEL_FILES:
            if not os.path.exists(fname):
                print(f"  ⚠ Skipping {fname} (not found locally)")
                continue
            size_mb = os.path.getsize(fname) / 1_048_576
            print(f"  ↑ {fname} ({size_mb:.1f} MB) ...", end=" ", flush=True)
            try:
                upload_file(
                    path_or_fileobj=fname,
                    path_in_repo=fname,
                    repo_id=SPACE_ID,
                    repo_type="space",
                    token=token,
                    commit_message=f"Upload model: {fname}",
                )
                print("✅")
            except Exception as e:
                print(f"❌ {e}")

    print(f"\n🌐 Space URL: https://huggingface.co/spaces/{SPACE_ID}")
    print("   (Build takes ~2 min after upload. Refresh the Space to see it live.)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True, help="HF write access token")
    parser.add_argument("--no-models", action="store_true",
                        help="Skip uploading large model files (demo mode only)")
    args = parser.parse_args()
    upload(args.token, include_models=not args.no_models)
