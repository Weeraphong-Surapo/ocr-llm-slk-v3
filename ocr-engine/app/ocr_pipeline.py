from __future__ import annotations

import io
import json
from typing import Any, Dict, Iterable, List

import fitz
import numpy as np
from PIL import Image

SUPPORTED_TYPES = {'application/pdf', 'image/png', 'image/jpeg', 'image/webp'}


def render_pdf_pages(data: bytes, max_pages: int) -> Iterable[np.ndarray]:
    document = fitz.open(stream=data, filetype='pdf')
    try:
        if document.page_count > max_pages:
            raise ValueError(f'PDF exceeds maximum page count of {max_pages}')
        for page in document:
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = np.frombuffer(pixmap.samples, dtype=np.uint8)
            yield image.reshape(pixmap.height, pixmap.width, pixmap.n)
    finally:
        document.close()


def decode_image(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as image:
        return np.asarray(image.convert('RGB'))


def result_to_dict(result: Any) -> Dict[str, Any]:
    value = getattr(result, 'json', None)
    if callable(value):
        value = value()
    if value is None:
        value = result
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError('PaddleOCR returned an unsupported result type')
    nested = value.get('res')
    return nested if isinstance(nested, dict) else value


def _list_value(data: Dict[str, Any], key: str) -> List[Any]:
    value = data.get(key, [])
    if hasattr(value, 'tolist'):
        value = value.tolist()
    return list(value or [])


def normalize_result(result: Any) -> Dict[str, Any]:
    data = result_to_dict(result)
    texts = _list_value(data, 'rec_texts')
    scores = _list_value(data, 'rec_scores')
    boxes = _list_value(data, 'rec_polys') or _list_value(data, 'dt_polys')
    lines = []
    for index, text in enumerate(texts):
        raw_score = scores[index] if index < len(scores) else None
        raw_box = boxes[index] if index < len(boxes) else None
        lines.append({
            'text': str(text),
            'confidence': float(raw_score) if raw_score is not None else None,
            'bbox': raw_box,
        })
    return {'text': '\n'.join(line['text'] for line in lines if line['text']), 'lines': lines}
