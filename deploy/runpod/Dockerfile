ARG BASE_IMAGE=nvcr.io/nvidia/tensorrt:24.05-py3
FROM ${BASE_IMAGE}

WORKDIR /workspace/YOLOZU

# Minimal OS deps for common Python wheels (pycocotools/opencv) + git for run records.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# YOLOZU intentionally keeps torch out of requirements; this image is for tooling + TRT/ORT exporters.
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r requirements.txt && \
    python3 -m pip install \
      onnxruntime \
      opencv-python-headless \
      pycocotools \
      cuda-python

ENV PYTHONPATH=/workspace/YOLOZU
ENV PYTHONUNBUFFERED=1

