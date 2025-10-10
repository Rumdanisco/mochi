FROM runpod/pytorch:3.10-2.1.0-118

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y git ffmpeg

# Install Python packages
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy handler and start script
COPY handler.py .
COPY start.sh .
RUN chmod +x start.sh

CMD ["./start.sh"]
