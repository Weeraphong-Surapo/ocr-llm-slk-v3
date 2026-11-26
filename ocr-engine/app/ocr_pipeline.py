from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass, replace
from statistics import median
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import fitz
import numpy as np
from PIL import Image

SUPPORTED_TYPES = {'application/pdf', 'image/png', 'image/jpeg', 'image/webp'}
PDF_SCALE = 2
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
            pixmap = page.get_pixmap(matrix=fitz.Matrix(PDF_SCALE, PDF_SCALE), alpha=False)
            image = np.frombuffer(pixmap.samples, dtype=np.uint8)
            yield image.reshape(pixmap.height, pixmap.width, pixmap.n)
    finally:
        document.close()


def pdf_page_count(data: bytes, max_pages: int) -> int:
    with fitz.open(stream=data, filetype='pdf') as document:
        if document.page_count > max_pages:
            raise ValueError(f'PDF exceeds maximum page count of {max_pages}')
        return document.page_count

def render_pdf_field(data: bytes, page_number: int, box: tuple[float, float, float, float]) -> np.ndarray:
    left, top, right, bottom = box
    margin = (bottom - top) / 3
    clip = fitz.Rect(*[point / PDF_SCALE for point in (left - margin, top - margin, right + margin, bottom + margin)])
    scale = 3 * PDF_SCALE
    with fitz.open(stream=data, filetype='pdf') as document:
        pixmap = document[page_number - 1].get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False)
        return np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)


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


@dataclass(frozen=True)
class _LayoutCell:
    y: float
    left: float
    right: float
    height: float
    text: str
    confidence: Optional[float] = None


_DECIMAL_CELL = re.compile(r'[+-]?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)\.[0-9]+')


def _column_runs(cells: List[_LayoutCell], edge: Callable[[_LayoutCell], float], height: float) -> Iterable[List[_LayoutCell]]:
    columns: List[List[_LayoutCell]] = []
    for cell in sorted(cells, key=edge):
        if not columns or edge(cell) - edge(columns[-1][0]) > height / 2:
            columns.append([])
        columns[-1].append(cell)

    for column in columns:
        run: List[_LayoutCell] = []
        for cell in sorted(column, key=lambda item: item.y):
            if run and cell.y - run[-1].y > height * 3:
                if len(run) >= 4:
                    yield run
                run = []
            run.append(cell)
        if len(run) >= 4:
            yield run


def _align_parallel_columns(cells: List[_LayoutCell]) -> List[_LayoutCell]:
    """Pair a complete label/value block when the two columns have consistently shifted baselines."""
    if not cells:
        return cells
    height = median(cell.height for cell in cells)
    if height <= 0:
        return cells

    values = [cell for cell in cells if _DECIMAL_CELL.fullmatch(cell.text.strip())]
    labels = [cell for cell in cells if any(character.isalpha() for character in cell.text)]
    label_runs = list(_column_runs(labels, lambda cell: cell.left, height))
    aligned = {}
    for value_run in _column_runs(values, lambda cell: cell.right, height):
        gap = median(second.y - first.y for first, second in zip(value_run, value_run[1:]))
        left_edge = min(cell.left for cell in value_run)
        candidates = []
        for label_run in label_runs:
            if len(label_run) != len(value_run) or max(cell.right for cell in label_run) >= left_edge:
                continue
            offsets = [value.y - label.y for value, label in zip(value_run, label_run)]
            shift = median(offsets)
            if height / 2 < abs(shift) < gap * 0.85 and max(abs(offset - shift) for offset in offsets) <= height / 3:
                candidates.append(label_run)
        # Equal lengths and a single candidate keep missing or ambiguous cells in their original positions.
        if len(candidates) == 1:
            aligned.update((value, replace(value, y=label.y)) for value, label in zip(value_run, candidates[0]))
    return [aligned.get(cell, cell) for cell in cells]


def paddle_layout_rows(lines: List[Dict[str, Any]], *, align: bool = True) -> List[List[_LayoutCell]]:
    positioned = []
    for line in lines:
        if not line['text'].strip():
            continue
        if not line.get('bbox'):
            return []
        xs, ys = zip(*line['bbox'])
        positioned.append(_LayoutCell((min(ys) + max(ys)) / 2, min(xs), max(xs), max(ys) - min(ys),
                                      line['text'], line.get('confidence')))

    rows = []
    cells = _align_parallel_columns(positioned) if align else positioned
    for cell in sorted(cells, key=lambda item: (item.y, item.left)):
        if not rows or cell.y - rows[-1][0].y > min(cell.height, rows[-1][0].height) / 2:
            rows.append([])
        rows[-1].append(cell)
    return rows


def paddle_layout_text(lines: List[Dict[str, Any]]) -> str:
    """Join text at the same height in left-to-right order to preserve table cells."""
    rows = paddle_layout_rows(lines)
    if not rows:
        return '\n'.join(line['text'] for line in lines if line['text'])
    return '\n'.join(' | '.join(cell.text for cell in sorted(row, key=lambda item: item.left)) for row in rows)


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
