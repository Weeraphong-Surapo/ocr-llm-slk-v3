from typing import Any, Dict, Optional
import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .engine import OcrEngine

app = FastAPI(title='OCR Engine', version='0.1.0')
engine = OcrEngine()


@app.on_event('startup')
def load_model() -> None:
    engine.load()


@app.get('/health')
def health() -> Dict[str, Any]:
    return {
        'status': 'ok',
        'model_ready': engine.ready,
        'model': 'PaddleOCR PP-OCRv5 mobile det + th_PP-OCRv5_mobile_rec',
        'device': engine.device,
        'model_error': engine.error,
    }


@app.get('/health/gpu')
def gpu_health() -> Dict[str, Any]:
    try:
        import paddle
        return {
            'available': bool(paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0),
            'paddle_version': paddle.__version__,
            'cuda_compiled': paddle.is_compiled_with_cuda(),
            'device_count': paddle.device.cuda.device_count(),
            'configured_device': os.getenv('CUDA_VISIBLE_DEVICES', 'all'),
            'ocr_device': engine.device,
            'model_ready': engine.ready,
        }
    except Exception as exc:
        return {
            'available': False,
            'model_ready': engine.ready,
            'error': f'{type(exc).__name__}: {exc}',
        }


@app.post('/ocr')
async def ocr(file: UploadFile = File(...), prompt: str = Form(...)) -> Dict[str, Any]:
    supported = {'application/pdf', 'image/png', 'image/jpeg', 'image/webp'}
    if file.content_type not in supported:
        raise HTTPException(status_code=422, detail='Unsupported file type')
    if not engine.ready:
        raise HTTPException(status_code=503, detail=engine.error or 'OCR model is not ready')
    data = await file.read()
    try:
        return engine.process(data, file.content_type or '')
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'OCR inference failed: {exc}') from exc
