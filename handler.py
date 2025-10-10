import os
import shutil
import runpod
import torch
from diffusers import DiffusionPipeline
from PIL import Image
from io import BytesIO
import base64
import psutil  # For monitoring disk space

# --- Cache directory setup ---
CACHE_DIR = "/workspace/.cache/huggingface"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)

# --- Function: clean cache if disk nearly full ---
def clean_cache(threshold_gb=30):
    """Automatically clear cache if disk usage > threshold_gb."""
    total, used, free = shutil.disk_usage("/workspace")
    used_gb = used / (1024**3)
    total_gb = total / (1024**3)

    print(f"💾 Disk usage: {used_gb:.1f} GB / {total_gb:.1f} GB total")

    if used_gb > threshold_gb:
        print("⚠️ Disk usage too high — clearing cache to free space...")
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
        os.makedirs(CACHE_DIR, exist_ok=True)
        print("✅ Cache cleared successfully.")

# --- Run cleanup before loading model ---
clean_cache(threshold_gb=30)

# --- Load Mochi-1 model (cached if available) ---
print("🚀 Loading Mochi-1 model (cached if already present)...")
pipe = DiffusionPipeline.from_pretrained(
    "genmo/mochi-1-preview",
    torch_dtype=torch.float16,
    cache_dir=CACHE_DIR
).to("cuda")
pipe.enable_model_cpu_offload()

# --- RunPod handler ---
def generate_video(job):
    """RunPod job handler for Mochi-1 Preview."""
    input_data = job["input"]
    prompt = input_data.get("prompt", "")
    image_b64 = input_data.get("image", None)

    print(f"🎬 Generating video for prompt: {prompt}")

    if image_b64:
        image = Image.open(BytesIO(base64.b64decode(image_b64)))
        result = pipe(prompt=prompt, image=image)
    else:
        result = pipe(prompt=prompt)

    # Save output video
    os.makedirs("/workspace/output", exist_ok=True)
    output_path = f"/workspace/output/output.mp4"
    result.videos[0].save(output_path)

    print("✅ Video saved:", output_path)
    return {"output_video": output_path}

# --- Start RunPod handler ---
runpod.serverless.start({"handler": generate_video})
