import os
import torch
import shutil
import psutil
from flask import Flask, request, jsonify, send_file
from diffusers import DiffusionPipeline
from PIL import Image
import io
import base64
import requests

# ─────────────────────────────
#  CONFIGURATION
# ─────────────────────────────
CACHE_DIR = os.getenv("HF_HOME", "/workspace/hf_cache")
MODEL_REPO = os.getenv("MODEL_REPO", "/workspace/mochi-1-preview")  # local path preferred
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────
#  CACHE CLEANUP
# ─────────────────────────────
def cleanup_cache(threshold_gb=8):
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024 ** 3)
    print(f"[INFO] Free disk space: {free_gb:.2f} GB")

    if free_gb < threshold_gb:
        print("[WARNING] Low disk space — clearing cache...")
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR, ignore_errors=True)
        os.makedirs(CACHE_DIR, exist_ok=True)

# ─────────────────────────────
#  IMAGE LOADING
# ─────────────────────────────
def load_image(image_input):
    """Accepts base64 string or URL."""
    if not image_input:
        return None
    try:
        if image_input.startswith("http"):
            response = requests.get(image_input)
            return Image.open(io.BytesIO(response.content)).convert("RGB")
        else:
            return Image.open(io.BytesIO(base64.b64decode(image_input))).convert("RGB")
    except Exception as e:
        print(f"[ERROR] Failed to load image: {e}")
        return None

# ─────────────────────────────
#  MODEL LOAD
# ─────────────────────────────
print(f"[INFO] Loading model from: {MODEL_REPO}")
pipe = DiffusionPipeline.from_pretrained(
    MODEL_REPO,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    use_safetensors=True,
    variant="fp16",
    cache_dir=CACHE_DIR
).to(DEVICE)

# ─────────────────────────────
#  GENERATE FUNCTION
# ─────────────────────────────
def generate_video(prompt, image_input=None, fps=24):
    cleanup_cache()

    if image_input:
        init_image = load_image(image_input)
        result = pipe(prompt=prompt, image=init_image)
    else:
        result = pipe(prompt=prompt)

    output_path = "/workspace/output.mp4"
    pipe.save_video(result.frames, output_path, fps=fps)
    print(f"[INFO] Video saved to: {output_path}")
    return output_path

# ─────────────────────────────
#  FLASK APP
# ─────────────────────────────
app = Flask(__name__)

@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "Mochi-1 Preview Video Generator API is running 🚀"})

@app.route("/generate", methods=["POST"])
def api_generate():
    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    image_input = data.get("image", None)

    if not prompt:
        return jsonify({"error": "Missing 'prompt' field"}), 400

    try:
        video_path = generate_video(prompt, image_input)
        return send_file(video_path, mimetype="video/mp4")
    except Exception as e:
        print(f"[ERROR] Generation failed: {e}")
        return jsonify({"error": str(e)}), 500

# ─────────────────────────────
#  RUN SERVER
# ─────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    print(f"[INFO] Starting server on port {port} ...")
    app.run(host="0.0.0.0", port=port)
