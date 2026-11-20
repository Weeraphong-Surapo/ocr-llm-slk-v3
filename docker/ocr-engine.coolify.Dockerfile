# CUDA 13.0 build for RTX 5060 Ti (Blackwell); host driver reports CUDA 13.2.
# No TensorRT variant, to keep the image small (target host has limited disk).
# paddleocr==3.0.3 with paddlepaddle 3.3.0 is untested: verify /health model_ready.
FROM paddlepaddle/paddle:3.3.0-gpu-cuda13.0-cudnn9.13
WORKDIR /app
COPY requirements.gpu.txt ./
RUN python -m pip install --no-cache-dir -r requirements.gpu.txt
COPY app ./app
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
