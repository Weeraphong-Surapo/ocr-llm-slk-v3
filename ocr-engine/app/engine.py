from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import closing
from typing import Any, Dict, List, Optional, Tuple

from .answer import INVALID_DECIMAL, AnswerClient, build_answer, interpret_item_cells, recover_decimal_fields
from .cancellation import OcrCancelled, check_cancelled
from .crosscheck import find_missing, find_suspicious
from .field_labels import identifier, verify_field_labels
from .item_table import TableRow, TableSpec, html_table_rows, item_table_crop, paddle_table_rows
from .ocr_pipeline import decode_image, normalize_result, paddle_layout_text, render_pdf_field, render_pdf_pages, scale_bbox, upscale_small_image
from .vlm import VlmClient, prepare_vlm_image

PADDLE_MODEL = 'PP-OCRv5_mobile_det + th_PP-OCRv5_mobile_rec'


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {'1', 'true', 'yes', 'on'}


class OcrEngine:
    def __init__(self, vlm: Optional[VlmClient] = None, llm: Optional[AnswerClient] = None) -> None:
        self.pipeline: Any = None
        self._identifier_recognizer: Any = None
        self._identifier_lock = threading.Lock()
        self.error: Optional[str] = None
        self.device = os.getenv('OCR_DEVICE', 'gpu:0')
        self.max_pages = int(os.getenv('OCR_MAX_PAGES', '50'))
        # Bound VLM requests within one document. The API queue remains one job
        # at a time, so two page requests do not multiply across users.
        self.page_concurrency = max(1, int(os.getenv('OCR_PAGE_CONCURRENCY', '2')))
        # Images whose long side is below this are enlarged before PaddleOCR (0 disables).
        self.upscale_min_long_side = int(os.getenv('OCR_UPSCALE_MIN_LONG_SIDE', '2000'))
        self.upscale_max_factor = float(os.getenv('OCR_UPSCALE_MAX_FACTOR', '4'))
        # Typhoon OCR via Ollama: main Thai reader. PaddleOCR cross-checks its output.
        self.vlm_enabled = _env_bool('VLM_ENABLED', True)
        self.vlm_mode = os.getenv('VLM_MODE', 'selective').strip().lower()
        self.vlm_selective_min_confidence = float(os.getenv('VLM_SELECTIVE_MIN_CONFIDENCE', '0.88'))
        self.vlm_selective_min_chars = int(os.getenv('VLM_SELECTIVE_MIN_CHARS', '80'))
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
                # VLM receives one rendered page, unlike the answer LLM which
                # receives the complete OCR document and keeps a 16K window.
                num_predict=int(os.getenv('VLM_NUM_PREDICT', '4096')),
                num_ctx=int(os.getenv('VLM_NUM_CTX', '8192')),
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
                # Structured extraction from multi-page invoices can exceed 4K tokens;
                # leave a usable default while retaining an env override for deployments.
                num_predict=int(os.getenv('LLM_NUM_PREDICT', '8192')),
                keep_alive=os.getenv('LLM_KEEP_ALIVE', '-1'),
            )
        self.llm_ready = False
        self.llm_error: Optional[str] = None if self.llm else 'LLM disabled'
        self._llm_loaded = threading.Event()
        self._llm_lock = threading.Lock()
        self.llm_lazy_load = _env_bool('LLM_LAZY_LOAD', False)
        self._executor = ThreadPoolExecutor(max_workers=self.page_concurrency, thread_name_prefix='vlm')

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
        if self.llm and not self.llm_lazy_load:
            self._load_llm()
        elif self.llm:
            self.llm_error = 'lazy'
            self._llm_loaded.set()

    def _ensure_local_llm(self) -> None:
        if self.llm is None or self.llm_ready:
            return
        with self._llm_lock:
            if self.llm_ready:
                return
            try:
                self.llm_error = 'loading'
                self.llm.ensure_model()
                self.llm_ready = True
                self.llm_error = None
            except Exception as exc:
                self.llm_error = f'{type(exc).__name__}: {exc}'
                raise

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

    def _read_with_vlm(self, image: Any, cancelled: threading.Event, *, detail: bool = False) -> Dict[str, Any]:
        assert self.vlm is not None
        check_cancelled(cancelled)
        min_side = self.vlm_max_long_side if detail else self.vlm_min_long_side
        png, _ = prepare_vlm_image(image, min_side, self.vlm_max_long_side)
        check_cancelled(cancelled)
        markdown, reason = self.vlm.read_page(png, cancelled=cancelled)
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

    def _should_use_vlm(self, lines: List[Dict[str, Any]], paddle_text: str, is_pdf: bool) -> bool:
        """Use the expensive VLM only for weak PDF pages in selective mode."""
        if not is_pdf or self.vlm_mode == 'always':
            return True
        if self.vlm_mode in {'off', 'disabled'}:
            return False
        confidences = [float(line['confidence']) for line in lines if line.get('confidence') is not None]
        average_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return (not lines or len(paddle_text.strip()) < self.vlm_selective_min_chars or
                average_confidence < self.vlm_selective_min_confidence)

    def _wait_for_model(self, loaded: threading.Event, cancelled: threading.Event) -> None:
        deadline = time.monotonic() + self.vlm_wait_seconds
        while not loaded.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0 or cancelled.wait(min(remaining, 0.1)):
                break
        check_cancelled(cancelled)

    def process(self, data: bytes, content_type: str, prompt: str = '', response_format: str = 'auto',
                cancelled: Optional[threading.Event] = None,
                response_schema: Optional[Dict[str, Any]] = None, mode: str = 'standard') -> Dict[str, Any]:
        cancelled = cancelled if cancelled is not None else threading.Event()
        check_cancelled(cancelled)
        if not self.pipeline:
            raise RuntimeError(self.error or 'OCR model is not ready')
        if mode not in {'fast', 'standard'}:
            raise ValueError('mode must be fast or standard')
        if mode != 'fast' and self.vlm is not None and not self.vlm_ready and not self._vlm_loaded.is_set():
            self._wait_for_model(self._vlm_loaded, cancelled)
        use_vlm = mode != 'fast' and self.vlm is not None and self.vlm_ready
        if mode == 'standard' and self.llm is not None and self.llm_lazy_load:
            self._ensure_local_llm()
        answer_client = self.llm if mode == 'standard' else None
        answer_ready = self.llm_ready if mode == 'standard' else False
        answer_error = self.llm_error
        pages, review = self._read_pages(data, content_type, use_vlm, cancelled)
        check_cancelled(cancelled)
        text = '\n\n'.join(page['text'] for page in pages if page['text'])
        result: Dict[str, Any] = {
            'pages': pages,
            'text': text,
            'review': review,
            'engines': {'ocr': PADDLE_MODEL, 'vlm': self.vlm.model if use_vlm and self.vlm else None,
                        'llm': answer_client.model if answer_client else None},
            'mode': mode,
        }
        if prompt.strip() and answer_client is not None:
            prompt = prompt.strip()
            if mode == 'standard' and not self.llm_ready and not self.llm_lazy_load and not self._llm_loaded.is_set():
                self._wait_for_model(self._llm_loaded, cancelled)
            check_cancelled(cancelled)
            answer_text = text
            tables: list[list[TableRow]] = []
            if response_schema is not None:
                answer_text = '\n\n'.join(paddle_layout_text(page['lines']) if page['source'] == 'paddleocr'
                                          else page['text'] for page in pages if page['text'])
                spec = TableSpec.from_schema(response_schema)
                if spec is not None:
                    for page in pages:
                        if page['source'] == 'paddleocr':
                            tables.extend(paddle_table_rows(page['lines'], spec, page['page_number']))
                        elif all(item['reason'] != 'vlm_output_truncated' for item in page['review']):
                            tables.extend(html_table_rows(page['text'], spec, page['page_number']))
            answer = build_answer(answer_client, answer_ready, answer_error, prompt, response_format,
                                  answer_text, review, cancelled=cancelled, response_schema=response_schema, item_tables=tables)
            if (response_schema is not None and use_vlm and self.vlm_mode == 'selective'
                    and content_type == 'application/pdf' and len(pages) == 1
                    and any(item['reason'] == 'vlm_skipped_selective' for item in pages[0]['review'])
                    and answer['error'] is None and any(item['status'] == INVALID_DECIMAL for item in answer['review'])):
                note = self._retry_decimals(data, prompt, response_format, response_schema, answer,
                                            answer_client,
                                            pages[0]['lines'], tables, cancelled)
                pages[0]['review'].append(note)
                review.append(note)
            if response_schema is not None and answer['error'] is None:
                reread = None
                if content_type == 'application/pdf':
                    reread = lambda page_number, box: self._reread_identifier(data, page_number, box, cancelled)
                verify_field_labels(answer, response_schema, pages, reread)
            result['answer'] = answer
        check_cancelled(cancelled)
        return result

    def _reread_identifier(self, data: bytes, page_number: int, box: tuple[float, float, float, float],
                           cancelled: threading.Event) -> str | None:
        check_cancelled(cancelled)
        try:
            image = render_pdf_field(data, page_number, box)
            with self._identifier_lock:
                check_cancelled(cancelled)
                if self._identifier_recognizer is None:
                    from paddleocr import TextRecognition

                    self._identifier_recognizer = TextRecognition(
                        model_name='en_PP-OCRv5_mobile_rec', device=self.device, enable_mkldnn=False,
                    )
                check_cancelled(cancelled)
                prediction = next(iter(self._identifier_recognizer.predict(image)), None)
        except OcrCancelled:
            raise
        except Exception:
            return None
        check_cancelled(cancelled)
        if prediction is None or prediction['rec_score'] < 0.9:
            return None
        value = prediction['rec_text'].strip()
        return value if identifier(value) else None

    def _retry_decimals(self, data: bytes, prompt: str, response_format: str, schema: Dict[str, Any],
                        answer: Dict[str, Any], answer_client: Any, lines: list[dict[str, Any]], tables: list[list[TableRow]],
                        cancelled: threading.Event) -> Dict[str, Any]:
        note = {'page_number': 1, 'text': None, 'reason': 'vlm_decimal_retry', 'similarity': None}
        spec = TableSpec.from_schema(schema)
        crop = item_table_crop(answer['content'].get('items', []), tables, spec) if spec is not None else None
        try:
            check_cancelled(cancelled)
            with closing(render_pdf_pages(data, 1)) as images:
                image = next(images)
                if crop is not None:
                    left, top, right, bottom = crop
                    height, width = image.shape[:2]
                    if left < min(width, right) and top < min(height, bottom):
                        image = image[top:bottom, left:right]
                    else:
                        crop = None
                vision = self._read_with_vlm(image, cancelled, detail=crop is not None)
        except OcrCancelled:
            raise
        except Exception as exc:
            return {**note, 'reason': 'vlm_failed', 'detail': f'{type(exc).__name__}: {exc}'[:300]}
        if vision['done_reason'] == 'length':
            return {**note, 'reason': 'vlm_output_truncated'}
        markdown = vision['markdown']
        retry_tables = html_table_rows(markdown, spec, 1) if spec is not None else []
        retry_text = markdown
        if crop is not None:
            left, top, right, bottom = crop
            outside = [line for line in lines if not line['bbox']
                       or not all(left <= x < right and top <= y < bottom for x, y in line['bbox'])]
            retry_text = f'{paddle_layout_text(outside)}\n\n<reread_table>\n{markdown}\n</reread_table>'
        retry = build_answer(answer_client, bool(answer_client), None, prompt, response_format, retry_text,
                             [], cancelled=cancelled, response_schema=schema, item_tables=retry_tables)
        if spec is not None and answer_client is not None:
            interpret_item_cells(answer_client, prompt, retry_text, schema, answer, retry, spec, retry_tables, cancelled)
        recover_decimal_fields(answer, retry, markdown, schema, tables, retry_tables)
        return note

    def _read_pages(self, data: bytes, content_type: str, use_vlm: bool,
                    cancelled: threading.Event) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        pages: List[Dict[str, Any]] = []
        review: List[Dict[str, Any]] = []
        is_pdf = content_type == 'application/pdf'
        inputs = render_pdf_pages(data, self.max_pages) if is_pdf else [decode_image(data)]
        page_jobs = []
        try:
            for page_number, image in enumerate(inputs, start=1):
                # Keep PaddleOCR sequential while overlapping the independent Typhoon page calls.
                check_cancelled(cancelled)
                lines = self._read_with_paddle(image, upscale=not is_pdf)
                check_cancelled(cancelled)
                paddle_text = '\n'.join(line['text'] for line in lines if line['text'])
                use_vlm_for_page = use_vlm and self._should_use_vlm(lines, paddle_text, is_pdf)
                vlm_future = self._executor.submit(self._read_with_vlm, image, cancelled) if use_vlm_for_page else None
                page_jobs.append((page_number, lines, paddle_text, vlm_future, use_vlm_for_page))

            for page_number, lines, paddle_text, vlm_future, use_vlm_for_page in page_jobs:
                check_cancelled(cancelled)
                page_review: List[Dict[str, Any]] = []
                markdown: Optional[str] = None
                if vlm_future is not None:
                    try:
                        vlm_result = vlm_future.result()
                        check_cancelled(cancelled)
                        markdown = vlm_result['markdown']
                        if vlm_result['done_reason'] == 'length':
                            page_review.append({'text': None, 'reason': 'vlm_output_truncated', 'similarity': None})
                        if markdown is not None:
                            page_review.extend(find_suspicious(markdown, paddle_text, self.review_min_similarity))
                            page_review.extend(find_missing(markdown, lines, self.missing_max_similarity))
                    except OcrCancelled:
                        raise
                    except Exception as exc:
                        page_review.append({'text': None, 'reason': 'vlm_failed', 'similarity': None,
                                            'detail': f'{type(exc).__name__}: {exc}'[:300]})
                elif self.vlm is not None and not use_vlm:
                    page_review.append({'text': None, 'reason': 'vlm_unavailable', 'similarity': None,
                                        'detail': self.vlm_error})
                elif self.vlm is not None and not use_vlm_for_page:
                    page_review.append({'text': None, 'reason': 'vlm_skipped_selective', 'similarity': None,
                                        'detail': 'PaddleOCR confidence/text length passed the selective threshold'})
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
        finally:
            # Remove pending calls and wait for active ones to finish cleaning up.
            wait([future for _, _, _, future, _ in page_jobs if future is not None and not future.cancel()])
        return pages, review
