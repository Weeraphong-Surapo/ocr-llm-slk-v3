from __future__ import annotations

import io
import json
from typing import Any, Dict, Iterable, List, Optional, Tuple

import fitz
import numpy as np
from PIL import Image

SUPPORTED_TYPES = {'application/pdf', 'image/png', 'image/jpeg', 'image/webp'}
def sniff_file_type(data: bytes) -> Optional[str]:
    """Real file type from its first bytes; the declared content type is not trusted."""
    if data.startswith(b'%PDF-'):
        return 'application/pdf'
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'image/png'
    if data.startswith(b'\xff\xd8\xff'):
        return 'image/jpeg'
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return 'image/webp'
    return None


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
def upscale_small_image(image: np.ndarray, min_long_side: int, max_factor: float) -> Tuple[np.ndarray, float]:
    """Enlarge low-resolution images so small Thai glyphs (vowels, tone marks) stay legible.

    Returns the image to run OCR on and the factor applied (1.0 means unchanged).
    """
    height, width = image.shape[:2]
    long_side = max(height, width)
    if min_long_side <= 0 or long_side == 0 or long_side >= min_long_side:
        return image, 1.0
    factor = min(min_long_side / long_side, max_factor)
    if factor <= 1.0:
        return image, 1.0
    size = (round(width * factor), round(height * factor))
    resized = Image.fromarray(image).resize(size, Image.Resampling.LANCZOS)
    return np.asarray(resized), factor
def scale_bbox(bbox: Any, factor: float) -> Any:
    """Map a polygon from the upscaled image back to original-image pixels."""
    if bbox is None or factor == 1.0:
        return bbox
    return [[round(float(x) / factor), round(float(y) / factor)] for x, y in bbox]


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
            'bbox': raw_box.tolist() if hasattr(raw_box, 'tolist') else raw_box,
        })
    return {'text': '\n'.join(line['text'] for line in lines if line['text']), 'lines': lines}
