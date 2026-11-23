from threading import Event
from typing import Optional


class OcrCancelled(Exception):
    pass


def check_cancelled(cancelled: Optional[Event]) -> None:
    if cancelled is not None and cancelled.is_set():
        raise OcrCancelled('OCR job was cancelled')
