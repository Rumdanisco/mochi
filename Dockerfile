# Use a base image with PyTorch and CUDA preinstalled
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set up environment
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV MODEL_REPO="genmo/mochi-1-preview"
ENV HF_HOME="/workspace/hf_cache"
ENV TRANSFORMERS_CACHE="/workspace/hf_cache"
ENV HF_TOKEN=""
ENV PATH="/usr/local/bin:$PATH"

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip git git-lfs ffmpeg \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# Copy files
COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /workspace/requirements.txt

COPY handler.py /workspace/handler.py

# Default command
CMD ["python3", "handler.py"]
