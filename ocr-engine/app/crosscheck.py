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


def find_suspicious(vlm_markdown: str, paddle_text: str, min_similarity: float = 0.6) -> List[Dict[str, object]]:
    haystack = _compact(paddle_text)
    flags: List[Dict[str, object]] = []
    for word in document_words(vlm_markdown):
        if any(ch.isdigit() for ch in word) or '@' in word:
            if word not in haystack:
                flags.append({'text': word, 'reason': 'number_or_email_not_confirmed', 'similarity': None})
            continue
        similarity = _best_similarity(word, haystack)
        if similarity < min_similarity:
            flags.append({'text': word, 'reason': 'low_similarity', 'similarity': round(similarity, 2)})
    return flags
