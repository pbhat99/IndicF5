import torch
from huggingface_hub import hf_hub_download
from f5_tts.api import F5TTS
import traceback

REPO_ID = "ai4bharat/IndicF5"

print("Downloading model files...")
try:
    ckpt_file = hf_hub_download(repo_id=REPO_ID, filename="model.safetensors")
    vocab_file = hf_hub_download(repo_id=REPO_ID, filename="checkpoints/vocab.txt")
    print(f"ckpt_file: {ckpt_file}")
    print(f"vocab_file: {vocab_file}")
except Exception as e:
    print(f"Error downloading: {e}")
    exit(1)

print("Initializing F5TTS...")
try:
    f5tts = F5TTS(
        model_type="F5-TTS",
        ckpt_file=ckpt_file,
        vocab_file=vocab_file,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Success!")
except Exception as e:
    print("Failed to initialize F5TTS:")
    traceback.print_exc()
