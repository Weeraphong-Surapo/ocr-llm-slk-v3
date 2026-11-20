"""Flag words in the VLM output that PaddleOCR does not support.

The VLM (Typhoon) reads Thai far better than PaddleOCR but can invent plausible text
(e.g. a different province) or misread digits. PaddleOCR reads digits reliably and its
Thai, while noisy, is usually close to the real text. A VLM word with no similar text
in the PaddleOCR output, or a number/email PaddleOCR never saw, is marked for review.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, List

_FIGURE = re.compile(r'<figure>.*?</figure>', re.S)
_PAGE_NO = re.compile(r'<page_number>.*?</page_number>', re.S)
_TAG = re.compile(r'<[^>]+>')
_MARKDOWN = re.compile(r'[#*|`>☐☑]|\\_')
_EDGE = '()[]{}:;,"\'“”‘’.'
_WS = re.compile(r'\s+')


def _compact(text: str) -> str:
    return _WS.sub('', text)


def document_words(markdown: str) -> List[str]:
    """Words from the VLM markdown, ignoring figure descriptions and formatting."""
    text = _PAGE_NO.sub(' ', _FIGURE.sub(' ', markdown))
    text = _MARKDOWN.sub(' ', _TAG.sub(' ', text))
    words: List[str] = []
    seen = set()
    for raw in text.split():
        word = raw.strip(_EDGE)
        if len(word) >= 3 and word not in seen:
            seen.add(word)
            words.append(word)
    return words


def _best_similarity(word: str, haystack: str) -> float:
    if word in haystack:
        return 1.0
    size = len(word)
    best = 0.0
    matcher = SequenceMatcher(None, autojunk=False)
    matcher.set_seq2(word)
    for start in range(0, max(1, len(haystack) - size + 1)):
        matcher.set_seq1(haystack[start:start + size])
        if matcher.real_quick_ratio() <= best or matcher.quick_ratio() <= best:
            continue
        ratio = matcher.ratio()
        if ratio > best:
            best = ratio
            if best == 1.0:
                break
    return best


_CODE_TOKEN = re.compile(r'[A-Za-z0-9][A-Za-z0-9#/\-.]*')


def _digits(code: str) -> str:
    # OCR models swap letter O and digit 0 constantly; treat them as the same digit.
    return ''.join(ch for ch in code.replace('O', '0').replace('o', '0') if ch.isdigit())


def _code_confirmed(code: str, paddle_codes: List[str]) -> bool:
    """A product code like OF50016 is confirmed when the most similar PaddleOCR code has the same digits.

    PaddleOCR often misreads the letters (OR50016, 0F50024) but rarely the digits, so the digits
    carry the check; an exact letter match is not required.
    """
    if code in paddle_codes:
        return True
    # Any similar PaddleOCR code with the same digits confirms it. Picking only the single most
    # similar code fails on invoices full of near-identical codes (OF50010, OF50016, OR50016...).
    digits = _digits(code)
    return any(
        _digits(candidate) == digits and SequenceMatcher(None, code, candidate, autojunk=False).ratio() >= 0.6
        for candidate in paddle_codes
    )


def find_suspicious(vlm_markdown: str, paddle_text: str, min_similarity: float = 0.6) -> List[Dict[str, object]]:
    haystack = _compact(paddle_text)
    paddle_codes = _CODE_TOKEN.findall(paddle_text)
    flags: List[Dict[str, object]] = []
    for word in document_words(vlm_markdown):
        has_digit = any(ch.isdigit() for ch in word)
        if '@' in word or (has_digit and not any(ch.isalpha() for ch in word)):
            # Emails and pure numbers (amounts, tax IDs, dates) must match exactly.
            if word not in haystack:
                flags.append({'text': word, 'reason': 'number_or_email_not_confirmed', 'similarity': None})
            continue
        if has_digit:
            if not _code_confirmed(word, paddle_codes):
                flags.append({'text': word, 'reason': 'number_or_email_not_confirmed', 'similarity': None})
            continue
        similarity = _best_similarity(word, haystack)
        if similarity < min_similarity:
            flags.append({'text': word, 'reason': 'low_similarity', 'similarity': round(similarity, 2)})
    return flags


def _plain_vlm_text(markdown: str) -> str:
    return _compact(_MARKDOWN.sub(' ', _TAG.sub(' ', _PAGE_NO.sub(' ', _FIGURE.sub(' ', markdown)))))


def find_missing(vlm_markdown: str, paddle_lines: List[Dict[str, object]],
                 max_similarity: float = 0.4, min_confidence: float = 0.6, min_length: int = 6) -> List[Dict[str, object]]:
    """PaddleOCR lines with nothing similar in the VLM output: text the VLM probably skipped.

    PaddleOCR's Thai is noisy, so the threshold is low; only lines that barely
    resemble anything in the VLM text are reported.
    """
    haystack = _plain_vlm_text(vlm_markdown)
    flags: List[Dict[str, object]] = []
    for line in paddle_lines:
        text = str(line.get('text') or '')
        confidence = line.get('confidence')
        compact = _compact(text)
        if len(compact) < min_length or (confidence is not None and float(confidence) < min_confidence):
            continue
        similarity = _best_similarity(compact, haystack)
        if similarity < max_similarity:
            flags.append({'text': text, 'reason': 'missing_in_vlm', 'similarity': round(similarity, 2),
                          'bbox': line.get('bbox')})
    return flags
