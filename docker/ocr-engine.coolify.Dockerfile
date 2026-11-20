# NVIDIA/CUDA base is selected from the Paddle compatibility matrix.
# Validate this exact image and RTX 5060 Ti support on the target Ubuntu host.
FROM paddlepaddle/paddle:3.0.0-gpu-cuda12.6-cudnn9.5-trt10.5
WORKDIR /app
COPY requirements.gpu.txt ./
RUN python -m pip install --no-cache-dir -r requirements.gpu.txt
COPY app ./app
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
