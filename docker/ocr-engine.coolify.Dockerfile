# CUDA 13.0 build for RTX 5060 Ti (Blackwell); host driver reports CUDA 13.2.
# No TensorRT variant, to keep the image small (target host has limited disk).
# paddleocr 3.7.0 + paddlepaddle 3.3.0: verify /health model_ready after deploy.
FROM paddlepaddle/paddle:3.3.0-gpu-cuda13.0-cudnn9.13
WORKDIR /app
COPY requirements.gpu.txt ./
RUN python -m pip install --no-cache-dir -r requirements.gpu.txt
COPY app ./app
ENV PYTHONUNBUFFERED=1
# The base image puts cuda-13.0/compat (libcuda 580) first on LD_LIBRARY_PATH.
# On a host with a newer driver (595) that causes cuInit error 803 and Paddle sees 0 GPUs.
# Drop compat so the host driver's libcuda (injected by nvidia-container-toolkit) is used.
ENV LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/x86_64-linux/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
