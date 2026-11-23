from concurrent.futures import ThreadPoolExecutor
from threading import Event
from uuid import uuid4

import numpy as np
from fastapi.testclient import TestClient

from app import main
from app.engine import OcrEngine
from tests.test_vlm_pipeline import FakePipeline


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

        def read_page(self, png):
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
