FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /workspace
RUN apt-get update && apt-get install -y git curl libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy scipy opencv-python-headless tifffile zarr fsspec aiohttp requests safetensors huggingface_hub pillow matplotlib

COPY code/ /workspace/code/
COPY reproduce_pipeline.sh /workspace/reproduce_pipeline.sh
RUN chmod +x /workspace/reproduce_pipeline.sh

ENTRYPOINT ["/workspace/reproduce_pipeline.sh"]
