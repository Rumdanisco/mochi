import os
import torch
import psutil
import shutil
from flask import Flask, request, jsonify, send_file
from diffusers import DiffusionPipeline
from PIL import Image
import base64
import io
import requests

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
CACHE_DIR = os.getenv("HF_HOME", "/workspace/hf_cache")
MODEL_PATH = os.getenv("MODEL_PATH", "/workspace/storage/mochi_model")

app = Flask(__name__)

pipe = None  # global pipeline cache


# -------------------------------------------------------------------
# UTILITIES
# -------------------------------------------------------------------

def cleanup_cache(threshold_gb=8):
    """Auto-cleans cache if free disk space is below threshold."""
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024 ** 3)
    print(f"[INFO] Free disk space: {free_gb:.2f} GB")

    if free_gb < threshold_gb:
        print("[WARNING] Low disk space — clearing cache...")
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR, ignore_errors=True)
        os.makedirs(CACHE_DIR, exist_ok=True)


def load_image(image_input):
    """Load image from URL or base64 string."""
    if image_input.startswith("http"):
        response = requests.get(image_input)
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    else:
        return Image.open(io.BytesIO(base64.b64decode(image_input))).convert("RGB")


def load_model():
    """Loads the Mochi model (cached globally)."""
    global pipe
    if pipe is None:
        print(f"[INFO] Loading model from: {MODEL_PATH}")
        pipe = DiffusionPipeline.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            cache_dir=CACHE_DIR
        ).to("cuda")
        print("[INFO] Model loaded successfully and ready.")
    return pipe


# -------------------------------------------------------------------
# API ENDPOINT
# -------------------------------------------------------------------

@app.route("/generate", methods=["POST"])
def generate_video():
    cleanup_cache()

    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    image_input = data.get("image", None)

    print(f"[INFO] Generating for prompt: {prompt}")
    pipe = load_model()

    if image_input:
        init_image = load_image(image_input)
        video_frames = pipe(prompt=prompt, image=init_image).frames
    else:
        video_frames = pipe(prompt=prompt).frames

    output_path = "/workspace/output.mp4"
    pipe.save_video(video_frames, output_path, fps=24)

    print(f"[INFO] Video saved at: {output_path}")
    return send_file(output_path, mimetype="video/mp4")


# -------------------------------------------------------------------
# LOCAL TEST MODE
# -------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(CACHE_DIR, exist_ok=True)
    print("[INFO] Starting Mochi Video Server on port 8000...")
    app.run(host="0.0.0.0", port=8000)
