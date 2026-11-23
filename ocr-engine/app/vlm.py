"""Typhoon OCR (vision-language model) client served by Ollama."""
from __future__ import annotations

import asyncio
import base64
import html
import io
from threading import Event
from typing import Any, Dict, Optional, Tuple, Union

import httpx
import numpy as np
from PIL import Image

from .cancellation import check_cancelled

# Typhoon OCR 1.5 only works with this exact prompt (see ollama.com/scb10x/typhoon-ocr1.5-3b).
TYPHOON_15_PROMPT = """Extract all text from the image.

Instructions:
- Only return the clean Markdown.
- Do not include any explanation or extra text.
- You must include all information on the page.

Formatting Rules:
- Tables: Render tables using <table>...</table> in clean HTML format.
- Equations: Render equations using LaTeX syntax with inline ($...$) and block ($$...$$).
- Images/Charts/Diagrams: Wrap any clearly defined visual areas (e.g. charts, diagrams, pictures) in:

<figure>
Describe the image's main elements (people, objects, text), note any contextual clues (place, event, culture), mention visible text and its meaning, provide deeper analysis when relevant (especially for financial charts, graphs, or documents), comment on style or architecture if relevant, then give a concise overall summary. Describe in Thai.
</figure>

- Page Numbers: Wrap page numbers in <page_number>...</page_number> (e.g., <page_number>14</page_number>).
- Checkboxes: Use ☐ for unchecked and ☑ for checked boxes."""

# Generation settings recommended by the model authors; low temperature limits invented text.
TYPHOON_15_OPTIONS = {'temperature': 0.1, 'top_p': 0.6, 'repeat_penalty': 1.1}


def prepare_vlm_image(image: np.ndarray, min_long_side: int, max_long_side: int) -> Tuple[bytes, float]:
    """Resize so the long side is within [min_long_side, max_long_side] and encode as PNG.

    Small images are enlarged (fewer invented words in tests), large scans are reduced
    (VLM time grows with pixel count). Returns PNG bytes and the factor applied.
    """
    height, width = image.shape[:2]
    long_side = max(height, width)
    factor = 1.0
    if long_side and long_side < min_long_side:
        factor = min_long_side / long_side
    elif long_side > max_long_side:
        factor = max_long_side / long_side
    pil = Image.fromarray(image)
    if factor != 1.0:
        pil = pil.resize((round(width * factor), round(height * factor)), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    pil.save(buffer, format='PNG')
    return buffer.getvalue(), factor


def parse_keep_alive(value: str) -> Union[int, str]:
    """Ollama reads a JSON number as seconds (-1 = keep loaded forever) but a string
    must carry a unit such as "30m"; a bare "-1" string is rejected with HTTP 400."""
    value = value.strip()
    try:
        return int(value)
    except ValueError:
        return value


class VlmClient:
    def __init__(self, url: str, model: str, timeout: float, num_predict: int, keep_alive: str,
                 num_ctx: int = 12288) -> None:
        self.url = url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.num_predict = num_predict
        self.keep_alive = parse_keep_alive(keep_alive)
        # Image tokens, prompt and output share this window. Ollama's default (4096) is used up
        # by a ~1800 px page (~2900 image tokens) and cuts the answer off.
        self.num_ctx = num_ctx

    def _post(self, path: str, payload: Dict[str, Any], timeout: float,
              cancelled: Optional[Event] = None) -> Dict[str, Any]:
        url = f'{self.url}{path}'
        if cancelled is None:
            response = httpx.post(url, json=payload, timeout=timeout, follow_redirects=True)
        else:
            response = asyncio.run(self._post_cancellable(url, payload, timeout, cancelled))
        if not response.is_success:
            raise RuntimeError(f'Ollama {path} returned HTTP {response.status_code}: {response.text[:300]}')
        return response.json()

    @staticmethod
    async def _post_cancellable(url: str, payload: Dict[str, Any], timeout: float, cancelled: Event) -> httpx.Response:
        check_cancelled(cancelled)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            check_cancelled(cancelled)
            request = asyncio.create_task(client.post(url, json=payload))
            try:
                while not request.done():
                    await asyncio.wait((request,), timeout=0.05)
                    check_cancelled(cancelled)
                return request.result()
            finally:
                # Wait for HTTPX to close the connection before releasing the OCR job.
                if not request.done():
                    request.cancel()
                await asyncio.gather(request, return_exceptions=True)

    def ensure_model(self, pull_timeout: float = 1800) -> None:
        """Pull the model if missing, then load it into VRAM so the first job is not slow."""
        try:
            self._post('/api/show', {'model': self.model}, timeout=30)
        except RuntimeError:
            self._post('/api/pull', {'model': self.model, 'stream': False}, timeout=pull_timeout)
        # Load with the same num_ctx as real requests; a different value makes Ollama reload the model.
        self._post('/api/generate', {'model': self.model, 'prompt': '', 'keep_alive': self.keep_alive,
                                     'options': {'num_ctx': self.num_ctx}}, timeout=300)

    def read_page(self, png: bytes, cancelled: Optional[Event] = None) -> Tuple[str, str]:
        """Return (markdown, done_reason). done_reason 'length' means the output was cut off."""
        data = self._post('/api/generate', {
            'model': self.model,
            'prompt': TYPHOON_15_PROMPT,
            'images': [base64.b64encode(png).decode()],
            'stream': False,
            'keep_alive': self.keep_alive,
            'options': {**TYPHOON_15_OPTIONS, 'num_predict': self.num_predict, 'num_ctx': self.num_ctx},
        }, timeout=self.timeout, cancelled=cancelled)
        # Typhoon escapes HTML entities in plain text (e.g. MOVING&amp;SERVICE).
        text = html.unescape(str(data.get('response', ''))).strip()
        if not text:
            raise RuntimeError('Typhoon OCR returned empty output')
        return text, str(data.get('done_reason', ''))
