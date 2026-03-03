from huggingface_hub import snapshot_download
import os

# Create models directory
os.makedirs("./models", exist_ok=True)

print("="*70)
print("DOWNLOADING HUGGINGFACE MODELS (Direct Method)")
print("="*70)

# ========================================
# MODEL 1: openai-community/roberta-base-openai-detector
# ========================================
print("\n1. Downloading: openai-community/roberta-base-openai-detector")
print("-"*70)

try:
    model1_path = snapshot_download(
        repo_id="openai-community/roberta-base-openai-detector",
        cache_dir="./models",
        local_dir="./models/openai-community-roberta-base-openai-detector",
        local_dir_use_symlinks=False
    )
    print(f"✓ Model downloaded successfully")
    print(f"✓ Location: {model1_path}")
    
except Exception as e:
    print(f"✗ Error: {e}")

# ========================================
# MODEL 2: roberta-base-openai-detector
# ========================================
print("\n2. Downloading: roberta-base-openai-detector")
print("-"*70)

try:
    model2_path = snapshot_download(
        repo_id="roberta-base-openai-detector",
        cache_dir="./models",
        local_dir="./models/roberta-base-openai-detector",
        local_dir_use_symlinks=False
    )
    print(f"✓ Model downloaded successfully")
    print(f"✓ Location: {model2_path}")
    
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "="*70)
print("DOWNLOAD COMPLETE")
print("="*70)
