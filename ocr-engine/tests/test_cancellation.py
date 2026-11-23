from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread
from uuid import uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main
from app.answer import AnswerClient
from app.engine import OcrEngine
from app.vlm import VlmClient
from tests.test_vlm_pipeline import FakePipeline, png_bytes


def test_cancel_stops_pending_vlm_pages_and_waits_for_the_active_call(monkeypatch):
    monkeypatch.setenv('OCR_PAGE_CONCURRENCY', '1')
    monkeypatch.setenv('VLM_MODE', 'always')
    monkeypatch.setenv('LLM_ENABLED', 'false')
    started = Event()
    all_pages_scheduled = Event()
    release = Event()
    calls = []

    class BlockingVlm:
        model = 'fake-vlm'

        def read_page(self, png, cancelled=None):
            calls.append(png)
            started.set()
            if not release.wait(5):
                raise RuntimeError('Test did not release VLM')
            return 'text', 'stop'

    def render_pages(data, limit):
        for _ in range(3):
            yield np.zeros((10, 10, 3), dtype=np.uint8)
        all_pages_scheduled.set()

    engine = OcrEngine(vlm=BlockingVlm())
    engine.pipeline = FakePipeline()
    engine.vlm_ready = True
    engine.vlm_min_long_side = 10
    monkeypatch.setattr('app.engine.render_pdf_pages', render_pages)
    monkeypatch.setattr(engine, 'load', lambda: None)
    monkeypatch.setattr(main, 'engine', engine)
    job_id = str(uuid4())
    cancel_url = f'/ocr/{job_id}/cancel'

    try:
        with TestClient(main.app) as client, ThreadPoolExecutor(max_workers=1) as requests:
            running = requests.submit(
                client.post, '/ocr',
                headers={'X-OCR-Job-Id': job_id},
                files={'file': ('invoice.pdf', b'%PDF-fixture', 'application/pdf')},
            )
            try:
                assert started.wait(3)
                assert all_pages_scheduled.wait(3)
                assert client.post(cancel_url).status_code == 202
                assert client.post(cancel_url).status_code == 202
                assert not running.done()
            finally:
                release.set()
            assert running.result(timeout=3).status_code == 409
            assert len(calls) == 1
            assert client.post(cancel_url).status_code == 404
    finally:
        engine._executor.shutdown()


@pytest.mark.parametrize('model_path', ['/api/generate', '/api/chat'])
def test_cancel_closes_inflight_ollama_connection(monkeypatch, model_path):
    started = Event()
    disconnected = Event()
    requested_paths = []

    class SlowOllama(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            requested_paths.append(self.path)
            started.set()
            # Ollama is still generating and has not sent even the response headers.
            self.connection.settimeout(5)
            if self.connection.recv(1) == b'':
                disconnected.set()

    monkeypatch.setenv('VLM_ENABLED', 'false')
    monkeypatch.setenv('LLM_ENABLED', 'false')
    with ThreadingHTTPServer(('127.0.0.1', 0), SlowOllama) as ollama:
        server_thread = Thread(target=ollama.serve_forever, kwargs={'poll_interval': 0.05})
        server_thread.start()
        url = f'http://127.0.0.1:{ollama.server_port}'
        engine = OcrEngine(
            vlm=VlmClient(url, 'vision', 10, 100, '-1') if model_path == '/api/generate' else None,
            llm=AnswerClient(url, 'answer', 10, 8192, 100, '-1') if model_path == '/api/chat' else None,
        )
        engine.pipeline = FakePipeline()
        engine.vlm_ready = engine.llm_ready = True
        engine.vlm_min_long_side = 10
        engine.upscale_min_long_side = 0
        monkeypatch.setattr(engine, 'load', lambda: None)
        monkeypatch.setattr(main, 'engine', engine)
        job_id = str(uuid4())
        try:
            with TestClient(main.app) as client, ThreadPoolExecutor(max_workers=1) as requests:
                running = requests.submit(
                    client.post, '/ocr',
                    headers={'X-OCR-Job-Id': job_id},
                    data={'prompt': 'Extract fields', 'response_format': 'json'},
                    files={'file': ('invoice.png', png_bytes(), 'image/png')},
                )
                assert started.wait(3)
                assert client.post(f'/ocr/{job_id}/cancel').status_code == 202
                assert disconnected.wait(2)
                assert running.result(timeout=2).status_code == 409
                assert client.post(f'/ocr/{job_id}/cancel').status_code == 404
                assert requested_paths == [model_path]
        finally:
            engine._executor.shutdown()
            ollama.shutdown()
            server_thread.join()
