"""Verify identifiers against schema-provided labels and page positions."""
from __future__ import annotations

import re
from dataclasses import dataclass, replace

from jsonschema import Draft202012Validator

from .ocr_pipeline import paddle_layout_rows

FIELD_LABELS_KEY = 'x-ocr-field-labels'
_IDENTIFIER = re.compile(r'[\w./-]+(?:[ \t-]+[0-9๐-๙]+)*')
_GROUPED_DIGITS = re.compile(r'[0-9๐-๙]+(?:[ \t-]+[0-9๐-๙]+)*')
_THAI_DIGITS = str.maketrans('๐๑๒๓๔๕๖๗๘๙', '0123456789')
_MIN_CONFIDENCE = 0.75


@dataclass(frozen=True)
class Candidate:
    text: str
    label: str
    page: int
    box: tuple[float, float, float, float] | None
    reread: bool = False

    def evidence(self):
        result = {'label_text': self.label, 'source_text': self.text, 'page_number': self.page}
        if self.reread:
            result['method'] = 'paddle_crop'
        return result


def _aliases(values):
    if not isinstance(values, list):
        return []
    return sorted({value.strip().casefold() for value in values if isinstance(value, str) and value.strip()},
                  key=len, reverse=True)


def _after_label(text, labels, *, joined_thai=True):
    for label in labels:
        if text.casefold().startswith(label):
            rest = text[len(label):]
            if not rest or rest[0] in ' :：/#(.' or rest[0].isdigit():
                return rest.lstrip(' :：#./')
            # Party headings may be printed adjacent to the name.
            if joined_thai and '\u0e00' <= label[-1] <= '\u0e7f' and '\u0e00' <= rest[0] <= '\u0e7f':
                return rest
    return None


def identifier(text):
    return _IDENTIFIER.fullmatch(text) and any(character.isdigit() for character in text)


def _value(text, field):
    if field.get('normalize_digits') is True and _GROUPED_DIGITS.fullmatch(text):
        return re.sub(r'[\s-]+', '', text.translate(_THAI_DIGITS))
    return text


def _within_section(cell, anchor):
    return 0 <= cell.y - anchor.y <= 15 * cell.height and abs(cell.left - anchor.left) <= 4 * cell.height


def _reliable(cell):
    return cell.confidence is None or cell.confidence >= _MIN_CONFIDENCE


def _nearby_value(label, cells):
    # A handwritten value may sit above the baseline of its printed label.
    nearby = [cell for cell in cells if cell is not label
              and -label.height / 4 <= cell.left - label.right <= 8 * label.height
              and abs(cell.y - label.y) <= 0.8 * max(cell.height, label.height)]
    return min(nearby, key=lambda cell: (cell.left, abs(cell.y - label.y)), default=None)


def _field_text(text, field):
    if field.get('normalize_digits') is True:
        digits = _GROUPED_DIGITS.match(text)
        if digits and (not text[digits.end():] or text[digits.end():].lstrip()[:1].isalpha()):
            # A tax ID may share its OCR line with the adjacent branch heading.
            return digits.group()
    return text


def _candidates(pages, field):
    labels = _aliases(field.get('labels'))
    scoped = _aliases(field.get('scoped_labels'))
    other_labels = _aliases(field.get('other_labels'))
    sections = _aliases(field.get('sections'))
    other_sections = _aliases(field.get('other_sections'))
    direct, in_section, rejected = {}, {}, set()
    for page in pages:
        cells = [cell for row in paddle_layout_rows(page['lines'], align=False) for cell in row]
        anchors = [(cell, True) for cell in cells if _reliable(cell) and _after_label(cell.text.strip(), sections) is not None]
        anchors += [(cell, False) for cell in cells if _reliable(cell) and _after_label(cell.text.strip(), other_sections) is not None]
        anchors.sort(key=lambda entry: entry[0].y, reverse=True)
        for cell in cells:
            if not _reliable(cell):
                continue
            text = cell.text.strip()
            rest = _after_label(text, labels)
            target = direct
            owner = True
            if rest is None:
                rest = _after_label(text, other_labels)
                owner = False
            if rest is None:
                rest = _after_label(text, scoped, joined_thai=False)
                if rest is None:
                    continue
                target = in_section
                owner = next((belongs for anchor, belongs in anchors if _within_section(cell, anchor)), None)
                if owner is None and field.get('reject_before_section') is True:
                    if any(belongs and anchor.y > cell.y and _within_section(anchor, cell) for anchor, belongs in anchors):
                        owner = False
                if owner is None:
                    continue
            box = None
            if not rest:
                value_cell = _nearby_value(cell, cells)
                if value_cell is None or not _reliable(value_cell):
                    continue
                rest = value_cell.text.strip()
                box = (value_cell.left, value_cell.y - value_cell.height / 2,
                       value_cell.right, value_cell.y + value_cell.height / 2)
            rest = _field_text(rest, field)
            if not identifier(rest):
                continue
            value = _value(rest, field)
            if owner:
                candidate = Candidate(rest, text, page['page_number'], box)
                # A separate value box provides a precise crop for a second reading.
                if value not in target or (box is not None and target[value].box is None):
                    target[value] = candidate
            else:
                rejected.add(value)
    return direct or in_section, rejected


def verify_field_labels(answer, schema, pages, reread=None):
    fields = schema.get(FIELD_LABELS_KEY)
    if not isinstance(fields, dict) or not isinstance(answer.get('content'), dict):
        return
    collected = []
    values_by_path = {}
    reread_values = {}
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
        original_values = set(accepted)
        if reread is not None and field.get('reread') is True and len(accepted) == 1:
            value, candidate = next(iter(accepted.items()))
            if candidate.box is not None:
                region = candidate.page, candidate.box
                if region not in reread_values:
                    reread_values[region] = reread(candidate.page, candidate.box)
                alternative = reread_values[region]
                if isinstance(alternative, str) and alternative.endswith(value) and validator.is_valid(alternative):
                    prefix = alternative[:-len(value)]
                    if re.fullmatch(r'[A-Za-z]{1,2}', prefix):
                        accepted = {alternative: replace(candidate, text=alternative, reread=True)}
        collected.append((path, field, row, key, accepted, rejected))
        values_by_path[path] = (field, original_values | accepted.keys() | rejected)

    for path, field, row, key, accepted, rejected in collected:
        original = row[key]
        if not isinstance(original, str) or not original.strip():
            continue
        value = _value(original.strip(), field)
        if value in accepted:
            answer.setdefault('field_evidence', {})['$.' + path] = accepted[value].evidence()
            continue
        status = 'field_label_mismatch' if accepted else 'field_label_unverified'
        shared_value = field.get('shared_value')
        wrong_field = value in rejected or any(
            path != other and (not shared_value or shared_value != other_field.get('shared_value'))
            and _value(original.strip(), other_field) in candidates
            for other, (other_field, candidates) in values_by_path.items())
        if wrong_field or (field.get('prefer_label') is True and len(accepted) == 1):
            correction = next(iter(accepted)) if len(accepted) == 1 else None
            row[key] = correction
            status = 'field_label_wrong_field'
            if correction is not None:
                answer.setdefault('field_evidence', {})['$.' + path] = accepted[correction].evidence()
                status = 'field_label_corrected'
        answer['review'].append({'path': '$.' + path, 'value': original, 'status': status})
