# ============================================================
# Base image with CUDA + PyTorch (Runtime optimized)
# ============================================================
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# ============================================================
# Environment setup
# ============================================================
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH="/workspace/storage/mochi_model"
ENV HF_HOME="/workspace/hf_cache"
ENV TRANSFORMERS_CACHE="/workspace/hf_cache"
ENV PATH="/usr/local/bin:$PATH"

WORKDIR /workspace

# ============================================================
# Install dependencies
# ============================================================
RUN apt-get update && apt-get install -y \
    python3 python3-pip git git-lfs ffmpeg curl \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Install Python packages
# ============================================================
COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /workspace/requirements.txt

# ============================================================
# Copy your code
# ============================================================
COPY handler.py /workspace/handler.py

# ============================================================
# Expose Flask port (for API access)
# ============================================================
EXPOSE 8000

# ============================================================
# Default startup command
# ============================================================
CMD ["python3", "handler.py"]
