from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .ocr_pipeline import decode_image, normalize_result, render_pdf_pages


class OcrEngine:
    def __init__(self) -> None:
        self.pipeline: Any = None
        self.error: Optional[str] = None
        self.device = os.getenv('OCR_DEVICE', 'gpu:0')
        self.max_pages = int(os.getenv('OCR_MAX_PAGES', '50'))

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

    @property
    def ready(self) -> bool:
        return self.pipeline is not None

    def process(self, data: bytes, content_type: str) -> Dict[str, Any]:
        if not self.pipeline:
            raise RuntimeError(self.error or 'OCR model is not ready')
        pages: List[Dict[str, Any]] = []
        inputs = render_pdf_pages(data, self.max_pages) if content_type == 'application/pdf' else [decode_image(data)]
        for page_number, image in enumerate(inputs, start=1):
            predictions = self.pipeline.predict(image)
            page_lines = []
            for prediction in predictions:
                page_lines.extend(normalize_result(prediction)['lines'])
            pages.append({
                'page_number': page_number,
                'text': '\n'.join(line['text'] for line in page_lines if line['text']),
                'lines': page_lines,
            })
        return {'pages': pages, 'text': '\n\n'.join(page['text'] for page in pages if page['text'])}
