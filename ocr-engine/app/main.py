import asyncio
import os
from contextlib import suppress
from threading import Event
from typing import Any, Dict, Optional
from uuid import UUID

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from .cancellation import OcrCancelled
from .engine import OcrEngine
from .ocr_pipeline import sniff_file_type
from .response_schema import parse_response_schema

app = FastAPI(title='OCR Engine', version='0.1.0')
engine = OcrEngine()
active_jobs: Dict[UUID, Event] = {}


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
        'vlm_enabled': engine.vlm is not None,
        'vlm_model': engine.vlm.model if engine.vlm else None,
        'vlm_ready': engine.vlm_ready,
        'vlm_error': engine.vlm_error,
        'llm_model': engine.llm.model if engine.llm else None,
        'llm_ready': engine.llm_ready,
        'llm_error': engine.llm_error,
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
async def ocr(request: Request, file: UploadFile = File(...), prompt: str = Form(''), response_format: str = Form('auto'),
              response_schema: Optional[str] = Form(None),
              job_id: Optional[UUID] = Header(None, alias='X-OCR-Job-Id')) -> Dict[str, Any]:
    if response_format not in ('auto', 'json', 'text'):
        raise HTTPException(status_code=422, detail='response_format must be auto, json or text')
    try:
        schema = parse_response_schema(response_schema, response_format)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not engine.ready:
        raise HTTPException(status_code=503, detail=engine.error or 'OCR model is not ready')
    data = await file.read()
    content_type = sniff_file_type(data)
    if content_type is None:
        raise HTTPException(status_code=422, detail='Unsupported file type (expected PDF, PNG, JPEG or WebP content)')
    cancelled = Event()
    if job_id is not None:
        if job_id in active_jobs:
            raise HTTPException(status_code=409, detail='OCR job is already running')
        active_jobs[job_id] = cancelled
    disconnect = asyncio.create_task(cancel_on_disconnect(request, cancelled))
    try:
        # Typhoon takes seconds per page; keep the event loop free so /health stays responsive.
        return await run_in_threadpool(engine.process, data, content_type, prompt, response_format, cancelled, schema)
    except OcrCancelled as exc:
        raise HTTPException(status_code=409, detail='OCR job was cancelled') from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'OCR inference failed: {exc}') from exc
    finally:
        cancelled.set()
        disconnect.cancel()
        if job_id is not None:
            active_jobs.pop(job_id)
        with suppress(asyncio.CancelledError):
            await disconnect


async def cancel_on_disconnect(request: Request, cancelled: Event) -> None:
    while (await request.receive())['type'] != 'http.disconnect':
        pass
    cancelled.set()


@app.post('/ocr/{job_id}/cancel', status_code=202)
async def cancel_ocr(job_id: UUID) -> Dict[str, str]:
    cancelled = active_jobs.get(job_id)
    if cancelled is None:
        raise HTTPException(status_code=404, detail='OCR job is not running')
    cancelled.set()
    return {'status': 'cancelling'}
