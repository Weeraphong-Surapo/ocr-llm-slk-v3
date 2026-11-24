"""Locate item cells using table headers, rather than searching the row for a number."""
from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass, replace
from decimal import Decimal
from html.parser import HTMLParser
from typing import Any

from .ocr_pipeline import paddle_layout_rows

TABLE_KEY = 'x-ocr-item-table'
PADDLE_CELL_MIN_CONFIDENCE = 0.75
_NUMBER = r'[+-]?(?:(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]*)?|\.[0-9]+)'
_CELL_NUMBER = re.compile(rf'({_NUMBER})(?:\s*[^\W\d_][^0-9.,+-]*)?')


def _label(text: str) -> str:
    return ' '.join(text.casefold().strip(' :\t\r\n').split())


def _header_matches(label: str, aliases: set[str]) -> bool:
    # A bilingual header can contain two spellings for the same column.
    return label in aliases or any(label.startswith(alias + ' ') and label[len(alias):].strip(' /-()') in aliases
                                   for alias in aliases)


def _anchor(text: str) -> str:
    return re.sub(r'\s+', '', text.casefold())


def cell_number(source: str) -> str | None:
    match = _CELL_NUMBER.fullmatch(source.strip())
    return match[1] if match else None


def same_number(value: Any, source: str) -> bool:
    number = cell_number(source)
    return (isinstance(value, str) and re.fullmatch(_NUMBER, value) is not None and number is not None
            and Decimal(value.replace(',', '')) == Decimal(number.replace(',', '')))


@dataclass(frozen=True)
class TableSpec:
    aliases: dict[str, set[str]]
    anchors: tuple[str, ...]
    derived: dict[str, str]

    @classmethod
    def from_schema(cls, schema: dict[str, Any]) -> TableSpec | None:
        spec = schema.get(TABLE_KEY)
        items = schema.get('properties', {}).get('items', {})
        if (not isinstance(spec, dict) or schema.get('type') != 'object'
                or not isinstance(items, dict) or items.get('type') != 'array'):
            return None
        row = items.get('items')
        if not isinstance(row, dict) or row.get('type') != 'object':
            return None
        fields = row.get('properties', {})
        columns, anchors, derived = spec.get('columns'), spec.get('anchors'), spec.get('derived', {})
        if (not isinstance(fields, dict) or not isinstance(columns, dict) or not columns
                or not isinstance(anchors, list) or not anchors
                or not isinstance(derived, dict)):
            return None
        if any(not isinstance(field, str) or field not in fields for field in anchors):
            return None
        for field in columns.keys() | derived.keys():
            definition = fields.get(field)
            if (not isinstance(definition, dict) or definition.get('x-ocr-decimal') is not True
                    or not isinstance(definition.get('type'), list) or set(definition['type']) != {'string', 'null'}
                    or not isinstance(definition.get('pattern'), str)):
                return None
        if (any(not isinstance(names, list) or not names or any(not isinstance(name, str) or not _label(name)
                                                               for name in names) for names in columns.values())
                or any(not isinstance(source, str) or source not in columns for source in derived.values())
                or columns.keys() & derived.keys()):
            return None
        return cls({field: {_label(name) for name in names} for field, names in columns.items()},
                   tuple(anchors), derived)

    def columns(self, headers: list[str]) -> dict[str, int]:
        labels = [_label(header) for header in headers]
        positions = {}
        for field, aliases in self.aliases.items():
            matches = [index for index, label in enumerate(labels) if _header_matches(label, aliases)]
            if len(matches) != 1:
                return {}
            positions[field] = matches[0]
        return positions if len(set(positions.values())) == len(positions) else {}


@dataclass(frozen=True)
class TableRow:
    text: str
    printed_number: int | None
    cells: dict[str, str]
    row: int
    page: int
    reader: str
    unreliable: frozenset[str] = frozenset()
    crop: tuple[int, int, int, int] | None = None

    def evidence(self, column: str, method: str, value: Any) -> dict[str, Any]:
        return {'source_text': self.cells[column], 'column': column, 'row': self.row, 'page_number': self.page,
                'reader': self.reader, 'method': method, 'value': value}


class _HtmlTables(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self.table: list[list[str]] = []
        self.row: list[str] | None = None
        self.cell: list[str] | None = None
        self.depth = 0
        self.usable = True
        self.colspan = 1

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == 'table':
            self.depth += 1
            self.usable = self.depth == 1
        elif self.depth == 1 and tag == 'tr':
            if self.row is not None:
                self.usable = False
            self.row = []
        elif self.depth == 1 and self.row is not None and tag in {'td', 'th'}:
            if self.cell is not None:
                self.usable = False
            attributes = dict(attrs)
            span = attributes.get('colspan', '1') or ''
            if attributes.get('rowspan', '1') != '1' or not span.isascii() or not span.isdigit() or len(span) > 2:
                self.usable = False
            else:
                self.colspan = max(1, int(span))
            self.cell = []
        elif self.cell is not None and tag in {'br', 'p', 'div'}:
            self.cell.append(' ')

    def handle_endtag(self, tag: str) -> None:
        if tag == 'table' and self.depth:
            self.depth -= 1
            if self.depth == 0:
                if self.usable:
                    self.tables.append(self.table)
                self.table, self.row, self.cell = [], None, None
        elif self.depth == 1 and self.row is not None:
            if tag in {'td', 'th'} and self.cell is not None:
                if self.colspan == 1:
                    self.row.append(' '.join(''.join(self.cell).split()))
                else:
                    self.row.extend([''] * self.colspan)
                self.cell = None
            elif tag == 'tr' and self.cell is None:
                self.table.append(self.row)
                self.row = None

    def handle_data(self, data: str) -> None:
        if self.depth == 1 and self.cell is not None:
            self.cell.append(data)


def _table_row(labels: list[str], cells: dict[str, str], row: int, page: int, reader: str) -> TableRow:
    labels = [label for label in labels if label]
    number = re.fullmatch(r'([0-9]{1,6})[.)]?', labels[0]) if labels else None
    return TableRow(_anchor(' '.join(labels)), int(number[1]) if number else None, cells, row, page, reader)


def html_table_rows(markdown: str, spec: TableSpec, page: int) -> list[list[TableRow]]:
    parser = _HtmlTables()
    parser.feed(markdown)
    tables = []
    for table in parser.tables:
        for header_index, header in enumerate(table):
            columns = spec.columns(header)
            if columns:
                break
        else:
            continue
        tables.append([
            _table_row([cell for index, cell in enumerate(cells) if index not in columns.values()],
                       {field: cells[index] for field, index in columns.items()}, row, page, 'vlm')
            for row, cells in enumerate(table[header_index + 1:], 1) if len(cells) == len(header)
        ])
    return [table for table in tables if table]


def paddle_table_rows(lines: list[dict[str, Any]], spec: TableSpec, page: int) -> list[list[TableRow]]:
    layout = paddle_layout_rows(lines, align=False)
    tables = []
    for index, header in enumerate(layout):
        header = sorted(header, key=lambda cell: cell.left)
        columns = spec.columns([cell.text for cell in header])
        if not columns:
            continue
        by_index = {position: field for field, position in columns.items()}
        centers = [(cell.left + cell.right) / 2 for cell in header]
        if len(centers) < 2 or any(first.right > second.left for first, second in zip(header, header[1:])):
            continue
        boundaries = [max(0, centers[0] - (centers[1] - centers[0]) / 2)]
        boundaries.extend((first + second) / 2 for first, second in zip(centers, centers[1:]))
        boundaries.append(centers[-1] + (centers[-1] - centers[-2]) / 2)
        left = min(cell.left for cell in header)
        top = min(cell.y - cell.height / 2 for cell in header)
        margin = max(cell.height for cell in header)
        rows = []
        for row, cells in enumerate(layout[index + 1:], 1):
            ordered = sorted(cells, key=lambda cell: cell.left)
            if spec.columns([cell.text for cell in ordered]):
                break
            labels = [cell.text for cell in ordered if cell.right <= boundaries[min(by_index)]]
            values: dict[str, list[str]] = {field: [] for field in columns}
            ambiguous = set()
            unreliable = set()
            for cell in ordered:
                first = bisect_right(boundaries, cell.left) - 1
                last = bisect_right(boundaries, cell.right) - 1
                if first == last and first in by_index:
                    field = by_index[first]
                    values[field].append(cell.text)
                    if cell.confidence is not None and cell.confidence < PADDLE_CELL_MIN_CONFIDENCE:
                        unreliable.add(field)
                elif first != last:
                    ambiguous.update(field for position, field in by_index.items() if first <= position <= last)
            if labels and any(values.values()):
                table_row = _table_row(labels, {field: ' '.join(value) for field, value in values.items()
                                                if field not in ambiguous}, row, page, 'paddleocr')
                bottom = max(cell.y + cell.height / 2 for cell in cells)
                rows.append(replace(table_row, unreliable=frozenset(unreliable),
                                    crop=(int(left - margin), int(top - margin),
                                          int(boundaries[-1] + margin), int(bottom + margin))))
        if rows:
            tables.append(rows)
    return tables


def numbered_table_size(tables: list[list[TableRow]]) -> int | None:
    if len(tables) == 1 and all(row.printed_number == number for number, row in enumerate(tables[0], 1)):
        return len(tables[0])
    return None


def match_rows(items: list[dict[str, Any]], tables: list[list[TableRow]], spec: TableSpec) -> list[TableRow | None]:
    rows = [row for table in tables for row in table]
    matched = []
    for item in items:
        anchors = [_anchor(item[field]) for field in spec.anchors if isinstance(item.get(field), str)]
        anchors = [anchor for anchor in anchors if len(anchor) >= 3]
        candidates = [[index for index, row in enumerate(rows) if anchor in row.text] for anchor in anchors]
        unique = {hits[0] for hits in candidates if len(hits) == 1}
        matched.append(unique)
    # When the output has no usable name/code, a complete numbered table gives an independent row order.
    if (numbered_table_size(tables) == len(items)
            and all(not candidates or candidates == {position} for position, candidates in enumerate(matched))):
        return rows
    matched = [next(iter(candidates)) if len(candidates) == 1 else None for candidates in matched]
    return [rows[index] if index is not None and matched.count(index) == 1 else None for index in matched]


def item_table_crop(items: list[dict[str, Any]], tables: list[list[TableRow]],
                    spec: TableSpec) -> tuple[int, int, int, int] | None:
    if len(tables) != 1 or not items:
        return None
    rows = match_rows(items, tables, spec)
    if any(row is None or row.crop is None for row in rows):
        return None
    left, top, right, bottom = zip(*(row.crop for row in rows))
    return max(0, min(left)), max(0, min(top)), max(right), max(bottom)
