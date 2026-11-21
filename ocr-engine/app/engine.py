from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from .answer import AnswerClient, build_answer
from .crosscheck import find_missing, find_suspicious
from .ocr_pipeline import decode_image, normalize_result, render_pdf_pages, scale_bbox, upscale_small_image
from .vlm import VlmClient, prepare_vlm_image

PADDLE_MODEL = 'PP-OCRv5_mobile_det + th_PP-OCRv5_mobile_rec'


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {'1', 'true', 'yes', 'on'}


class OcrEngine:
    def __init__(self, vlm: Optional[VlmClient] = None, llm: Optional[AnswerClient] = None) -> None:
        self.pipeline: Any = None
        self.error: Optional[str] = None
        self.device = os.getenv('OCR_DEVICE', 'gpu:0')
        self.max_pages = int(os.getenv('OCR_MAX_PAGES', '50'))
        # Images whose long side is below this are enlarged before PaddleOCR (0 disables).
        self.upscale_min_long_side = int(os.getenv('OCR_UPSCALE_MIN_LONG_SIDE', '2000'))
        self.upscale_max_factor = float(os.getenv('OCR_UPSCALE_MAX_FACTOR', '4'))
        # Typhoon OCR via Ollama: main Thai reader. PaddleOCR cross-checks its output.
        self.vlm_enabled = _env_bool('VLM_ENABLED', True)
        self.vlm_min_long_side = int(os.getenv('VLM_MIN_LONG_SIDE', '1300'))
        self.vlm_max_long_side = int(os.getenv('VLM_MAX_LONG_SIDE', '1800'))
        self.review_min_similarity = float(os.getenv('REVIEW_MIN_SIMILARITY', '0.6'))
        self.missing_max_similarity = float(os.getenv('MISSING_MAX_SIMILARITY', '0.4'))
        # While the VLM is still loading (e.g. right after a deploy), jobs wait this long before falling back.
        self.vlm_wait_seconds = float(os.getenv('VLM_WAIT_SECONDS', '120'))
        self.vlm = vlm
        if self.vlm is None and self.vlm_enabled:
            self.vlm = VlmClient(
                url=os.getenv('OLLAMA_URL', 'http://ollama:11434'),
                model=os.getenv('VLM_MODEL', 'scb10x/typhoon-ocr1.5-3b'),
                timeout=float(os.getenv('VLM_TIMEOUT_SECONDS', '120')),
                num_predict=int(os.getenv('VLM_NUM_PREDICT', '6144')),
                num_ctx=int(os.getenv('VLM_NUM_CTX', '16384')),
                keep_alive=os.getenv('VLM_KEEP_ALIVE', '-1'),
            )
        self.vlm_ready = False
        self.vlm_error: Optional[str] = None if self.vlm else 'VLM disabled'
        self._vlm_loaded = threading.Event()
        # Text LLM that answers the user's prompt from the OCR text.
        self.llm = llm
        if self.llm is None and _env_bool('LLM_ENABLED', True):
            self.llm = AnswerClient(
                url=os.getenv('OLLAMA_URL', 'http://ollama:11434'),
                model=os.getenv('LLM_MODEL', 'scb10x/typhoon2.5-qwen3-4b'),
                timeout=float(os.getenv('LLM_TIMEOUT_SECONDS', '120')),
                num_ctx=int(os.getenv('LLM_NUM_CTX', '16384')),
                num_predict=int(os.getenv('LLM_NUM_PREDICT', '4096')),
                keep_alive=os.getenv('LLM_KEEP_ALIVE', '-1'),
            )
        self.llm_ready = False
        self.llm_error: Optional[str] = None if self.llm else 'LLM disabled'
        self._llm_loaded = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='vlm')

    def load(self) -> None:
        try:
            if self.device.startswith('gpu'):
                import paddle
                if not paddle.is_compiled_with_cuda() or paddle.device.cuda.device_count() == 0:
                    raise RuntimeError(
                        f'OCR_DEVICE={self.device} but Paddle sees no CUDA GPU '
                        f'(compiled_with_cuda={paddle.is_compiled_with_cuda()}, '
                        f'device_count={paddle.device.cuda.device_count()}); refusing to fall back to CPU'
                    )
            from paddleocr import PaddleOCR
            self.pipeline = PaddleOCR(
                text_detection_model_name='PP-OCRv5_mobile_det',
                text_recognition_model_name='th_PP-OCRv5_mobile_rec',
                device=self.device,
                enable_mkldnn=False,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
            self.error = None
        except Exception as exc:
            self.pipeline = None
            self.error = f'{type(exc).__name__}: {exc}'
        if self.vlm or self.llm:
            # Pulling multi-GB models can take minutes; do not block startup or health checks.
            threading.Thread(target=self._load_models, name='model-load', daemon=True).start()

    def _load_models(self) -> None:
        if self.vlm:
            self._load_vlm()
        if self.llm:
            self._load_llm()

    def _load_llm(self, attempts: int = 20, wait_seconds: float = 15) -> None:
        assert self.llm is not None
        for attempt in range(1, attempts + 1):
            try:
                self.llm_error = 'loading'
                self.llm.ensure_model()
                self.llm_ready = True
                self.llm_error = None
                break
            except Exception as exc:
                self.llm_ready = False
                self.llm_error = f'{type(exc).__name__}: {exc} (attempt {attempt}/{attempts})'
                if attempt < attempts:
                    time.sleep(wait_seconds)
        self._llm_loaded.set()

    def _load_vlm(self, attempts: int = 20, wait_seconds: float = 15) -> None:
        """Pull and warm up the VLM, retrying because Ollama may still be starting."""
        assert self.vlm is not None
        for attempt in range(1, attempts + 1):
            try:
                self.vlm_error = 'loading'
                self.vlm.ensure_model()
                self.vlm_ready = True
                self.vlm_error = None
                self._vlm_loaded.set()
                return
            except Exception as exc:
                self.vlm_ready = False
                self.vlm_error = f'{type(exc).__name__}: {exc} (attempt {attempt}/{attempts})'
                if attempt < attempts:
                    time.sleep(wait_seconds)
        self._vlm_loaded.set()  # gave up: stop making jobs wait

    @property
    def ready(self) -> bool:
        return self.pipeline is not None

    def _read_with_vlm(self, image: Any) -> Dict[str, Any]:
        assert self.vlm is not None
        png, _ = prepare_vlm_image(image, self.vlm_min_long_side, self.vlm_max_long_side)
        markdown, reason = self.vlm.read_page(png)
        return {'markdown': markdown, 'done_reason': reason}

    def _read_with_paddle(self, image: Any, upscale: bool) -> List[Dict[str, Any]]:
        factor = 1.0
        if upscale:
            image, factor = upscale_small_image(image, self.upscale_min_long_side, self.upscale_max_factor)
        lines: List[Dict[str, Any]] = []
        for prediction in self.pipeline.predict(image):
            for line in normalize_result(prediction)['lines']:
                line['bbox'] = scale_bbox(line['bbox'], factor)
                lines.append(line)
        return lines

    def process(self, data: bytes, content_type: str, prompt: str = '', response_format: str = 'auto') -> Dict[str, Any]:
        if not self.pipeline:
            raise RuntimeError(self.error or 'OCR model is not ready')
        if self.vlm is not None and not self.vlm_ready and not self._vlm_loaded.is_set():
            self._vlm_loaded.wait(self.vlm_wait_seconds)
        use_vlm = self.vlm is not None and self.vlm_ready
        pages: List[Dict[str, Any]] = []
        review: List[Dict[str, Any]] = []
        is_pdf = content_type == 'application/pdf'
        inputs = render_pdf_pages(data, self.max_pages) if is_pdf else [decode_image(data)]
        for page_number, image in enumerate(inputs, start=1):
            # Typhoon runs over HTTP while PaddleOCR runs here, so the page costs about one Typhoon call.
            vlm_future = self._executor.submit(self._read_with_vlm, image) if use_vlm else None
            lines = self._read_with_paddle(image, upscale=not is_pdf)
            paddle_text = '\n'.join(line['text'] for line in lines if line['text'])
            page_review: List[Dict[str, Any]] = []
            markdown: Optional[str] = None
            if vlm_future is not None:
                try:
                    vlm_result = vlm_future.result()
                    markdown = vlm_result['markdown']
                    if vlm_result['done_reason'] == 'length':
                        page_review.append({'text': None, 'reason': 'vlm_output_truncated', 'similarity': None})
                    page_review.extend(find_suspicious(markdown, paddle_text, self.review_min_similarity))
                    page_review.extend(find_missing(markdown, lines, self.missing_max_similarity))
                except Exception as exc:
                    page_review.append({'text': None, 'reason': 'vlm_failed', 'similarity': None,
                                        'detail': f'{type(exc).__name__}: {exc}'[:300]})
            elif self.vlm is not None:
                page_review.append({'text': None, 'reason': 'vlm_unavailable', 'similarity': None,
                                    'detail': self.vlm_error})
            for item in page_review:
                item['page_number'] = page_number
            review.extend(page_review)
            pages.append({
                'page_number': page_number,
                # Typhoon markdown when available, otherwise PaddleOCR text (flagged in review).
                'text': markdown if markdown is not None else paddle_text,
                'source': 'vlm' if markdown is not None else 'paddleocr',
                'paddle_text': paddle_text,
                'lines': lines,
                'review': page_review,
            })
        text = '\n\n'.join(page['text'] for page in pages if page['text'])
        result: Dict[str, Any] = {
            'pages': pages,
            'text': text,
            'review': review,
            'engines': {'ocr': PADDLE_MODEL, 'vlm': self.vlm.model if use_vlm and self.vlm else None,
                        'llm': self.llm.model if self.llm else None},
        }
        if prompt.strip() and self.llm is not None:
            if not self.llm_ready and not self._llm_loaded.is_set():
                self._llm_loaded.wait(self.vlm_wait_seconds)
            result['answer'] = build_answer(self.llm, self.llm_ready, self.llm_error, prompt.strip(),
                                            response_format, text, review)
        return result
