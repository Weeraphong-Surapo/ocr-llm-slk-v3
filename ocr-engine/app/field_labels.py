"""Verify identifiers against schema-provided labels and page positions."""
from __future__ import annotations

import re

from jsonschema import Draft202012Validator

from .ocr_pipeline import paddle_layout_rows

FIELD_LABELS_KEY = 'x-ocr-field-labels'
_IDENTIFIER = re.compile(r'[\w./-]+(?:[ \t-]+[0-9๐-๙]+)*')
_THAI_DIGITS = str.maketrans('๐๑๒๓๔๕๖๗๘๙', '0123456789')


def _aliases(values):
    if not isinstance(values, list):
        return []
    return sorted({value.strip().casefold() for value in values if isinstance(value, str) and value.strip()},
                  key=len, reverse=True)


def _after_label(text, labels):
    for label in labels:
        if text.casefold().startswith(label):
            rest = text[len(label):]
            if not rest or rest[0] in ' :：/#(.' or rest[0].isdigit():
                return rest.lstrip(' :：#./')
            # Thai labels are often printed without a space before the name or number.
            if '\u0e00' <= label[-1] <= '\u0e7f' and '\u0e00' <= rest[0] <= '\u0e7f':
                return rest
    return None


def _value(text, field):
    if field.get('normalize_digits') is True:
        text = text.translate(_THAI_DIGITS)
        if re.fullmatch(r'[0-9]+(?:[\s-]+[0-9]+)*', text):
            return re.sub(r'[\s-]+', '', text)
    return text


def _candidates(pages, field):
    labels = _aliases(field.get('labels'))
    scoped = _aliases(field.get('scoped_labels'))
    other_labels = _aliases(field.get('other_labels'))
    sections = _aliases(field.get('sections'))
    other_sections = _aliases(field.get('other_sections'))
    accepted, rejected = {}, set()
    for page in pages:
        anchors = []
        for row in paddle_layout_rows(page['lines'], align=False):
            for index, cell in enumerate(row):
                text = cell.text.strip()
                rest = _after_label(text, labels)
                owner = True
                if rest is None:
                    rest = _after_label(text, other_labels)
                    owner = False
                if rest is None:
                    rest = _after_label(text, scoped)
                    if rest is not None:
                        owner = next((kind for anchor, kind in reversed(anchors)
                                      if 0 <= cell.y - anchor.y <= 15 * cell.height
                                      and abs(cell.left - anchor.left) <= 4 * cell.height), None)
                if rest is None:
                    if _after_label(text, sections) is not None:
                        anchors.append((cell, True))
                    elif _after_label(text, other_sections) is not None:
                        anchors.append((cell, False))
                    continue
                if owner is None:
                    continue
                value_cell = cell
                if not rest and index + 1 < len(row):
                    value_cell = row[index + 1]
                    if value_cell.left - cell.right <= 8 * cell.height:
                        rest = value_cell.text.strip()
                if any(item.confidence is not None and item.confidence < 0.75 for item in (cell, value_cell)):
                    continue
                if not _IDENTIFIER.fullmatch(rest) or not any(character.isdigit() for character in rest):
                    continue
                value = _value(rest, field)
                if owner:
                    accepted[value] = {'label_text': text, 'source_text': rest, 'page_number': page['page_number']}
                else:
                    rejected.add(value)
    return accepted, rejected


def verify_field_labels(answer, schema, pages):
    fields = schema.get(FIELD_LABELS_KEY)
    if not isinstance(fields, dict) or not isinstance(answer.get('content'), dict):
        return
    collected = []
    values_by_path = {}
    for path, field in fields.items():
        if not isinstance(field, dict):
            continue
        location = path.split('.')
        row = answer['content']
        definition = schema
        for index, part in enumerate(location):
            definition = definition.get('properties', {}).get(part, {}) if isinstance(definition, dict) else {}
            if index < len(location) - 1:
                row = row.get(part) if isinstance(row, dict) else None
        key = location[-1]
        if not isinstance(definition, dict) or definition.get('type') not in (['string', 'null'], ['null', 'string']):
            continue
        if not isinstance(row, dict) or key not in row:
            continue
        validator = Draft202012Validator(definition)
        if not validator.is_valid(None):
            continue
        accepted, rejected = _candidates(pages, field)
        accepted = {value: evidence for value, evidence in accepted.items() if validator.is_valid(value)}
        collected.append((path, field, row, key, accepted, rejected))
        values_by_path[path] = (field, accepted.keys() | rejected)

    for path, field, row, key, accepted, rejected in collected:
        original = row[key]
        if not isinstance(original, str) or not original.strip():
            continue
        value = _value(original.strip(), field)
        if value in accepted:
            answer.setdefault('field_evidence', {})['$.' + path] = accepted[value]
            continue
        status = 'field_label_mismatch' if accepted else 'field_label_unverified'
        wrong_field = value in rejected or any(
            path != other and _value(original.strip(), other_field) in candidates
            for other, (other_field, candidates) in values_by_path.items())
        if wrong_field:
            correction = next(iter(accepted)) if len(accepted) == 1 else None
            row[key] = correction
            status = 'field_label_wrong_field'
            if correction is not None:
                answer.setdefault('field_evidence', {})['$.' + path] = accepted[correction]
                status = 'field_label_corrected'
        answer['review'].append({'path': '$.' + path, 'value': original, 'status': status})
