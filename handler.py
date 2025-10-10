import os
import torch
import psutil
import shutil
from diffusers import DiffusionPipeline
from PIL import Image
import base64
import io
import requests

CACHE_DIR = os.getenv("HF_HOME", "/workspace/hf_cache")
MODEL_REPO = os.getenv("MODEL_REPO", "genmo/mochi-1-preview")

# Cleanup function
def cleanup_cache(threshold_gb=8):
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024 ** 3)
    print(f"[INFO] Free disk space: {free_gb:.2f} GB")
    if free_gb < threshold_gb:
        print("[WARNING] Low disk space — clearing cache...")
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
        os.makedirs(CACHE_DIR, exist_ok=True)

def load_image(image_input):
    if image_input.startswith("http"):
        response = requests.get(image_input)
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    else:
        return Image.open(io.BytesIO(base64.b64decode(image_input))).convert("RGB")

def generate_video(input_data):
    cleanup_cache()  # Auto clean if low space

    prompt = input_data.get("prompt", "")
    image_input = input_data.get("image", None)

    print(f"[INFO] Loading model: {MODEL_REPO}")
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_REPO,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        cache_dir=CACHE_DIR
    ).to("cuda")

    if image_input:
        init_image = load_image(image_input)
        video_frames = pipe(prompt=prompt, image=init_image).frames
    else:
        video_frames = pipe(prompt=prompt).frames

    output_path = "/workspace/output.mp4"
    pipe.save_video(video_frames, output_path, fps=24)
    print(f"[INFO] Video saved at: {output_path}")

    return output_path

if __name__ == "__main__":
    # Test locally (you can change prompt)
    test_input = {
        "prompt": "A magical forest glowing at dusk with fireflies and mist"
    }
    generate_video(test_input)
