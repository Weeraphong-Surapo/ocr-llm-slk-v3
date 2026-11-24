import io
import json

import fitz
import pytest
from PIL import Image

from app.answer import AnswerClient
from app.engine import OcrEngine


@pytest.mark.parametrize(('amount', 'quantity', 'source', 'html', 'focused', 'corrected_name', 'unresolved'), [
    ('0', '1', 'ไม่มีค่าบริการ', True, None, False, set()),
    ('2.50', '1', 'ไม่มีค่าบริการ', True, None, False, set()),
    ('0', '18.69', 'ไม่มีค่าบริการ', True, None, False, {'quantity'}),
    ('0', '1', '18.69', True, {'value': '0', 'source': '18.69'}, False, {'amount', 'net_amount'}),
    ('0', '1', 'ไม่มีค่าบริการ', False, None, False, {'quantity', 'amount', 'net_amount'}),
    ('3.50', '1', 'ไม่มีค่าบริการ', True, {'value': '3.50', 'source': 'ไม่มีค่าบริการ'}, False, set()),
    ('0', '1', 'ไม่มีค่าบริการ', True, None, True, set()),
])
def test_selective_retry_requires_the_right_item_cell(amount, quantity, source, html, focused, corrected_name, unresolved):
    decimal = {'type': ['string', 'null'], 'pattern': r'^[0-9]+(?:\.[0-9]+)?$', 'x-ocr-decimal': True}
    fields = {'name': {'type': 'string'}, **dict.fromkeys(['quantity', 'price', 'amount', 'net_amount'], decimal)}
    schema = {
        'type': 'object', 'required': ['items'], 'properties': {'items': {'type': 'array', 'items': {
            'type': 'object', 'required': list(fields), 'properties': fields, 'additionalProperties': False,
        }}},
        'x-ocr-item-table': {
            'columns': {'quantity': ['QUANTITY', 'จำนวน'], 'price': ['UNIT PRICE', 'ราคาต่อหน่วย'],
                        'amount': ['AMOUNT', 'จำนวนเงิน']},
            'anchors': ['name'], 'derived': {'net_amount': 'amount'},
        },
    }
    paddle_name = 'Product X' if corrected_name else 'Product A'
    first = [
        {'name': paddle_name, 'quantity': '18.69', 'price': '18.69', 'amount': 'อ3', 'net_amount': 'อ3'},
        {'name': 'Product B', 'quantity': '22', 'price': '10', 'amount': '10', 'net_amount': '10'},
    ]
    retry = [
        {'name': 'Product A', 'quantity': quantity, 'price': '17.00',
         'amount': {'value': amount, 'source': source}, 'net_amount': amount},
        {**first[1], 'quantity': '1'},
    ]
    # Without a recognized reread table, even a valid number from the same row cannot fill a marked cell.
    if not html:
        retry[0]['amount'] = amount
    if focused and source == 'ไม่มีค่าบริการ':
        retry[0].update(amount=source, net_amount=None)
    responses = iter(({'items': first}, {'items': retry}, {'cell_1': focused, 'cell_2': focused}))
    requests = []

    def post(path, payload, timeout, cancelled=None):
        requests.append(payload)
        return {'message': {'content': json.dumps(next(responses))}, 'done_reason': 'stop'}

    class Paddle:
        def predict(self, image):
            rows = [
                ('No.', 'Description', 'QUANTITY', 'UNIT PRICE', 'AMOUNT'),
                ('?' if corrected_name else '1', paddle_name, '1 ชิ้น', '18.69', 'อ3'),
                ('2', 'Product B', '22', '10', '10'),
            ]
            boxes = []
            for top in (50, 76, 102):
                for left, right in ((32, 55), (74, 172), (218, 279), (318, 373), (411, 460)):
                    boxes.append([[left, top], [right, top], [right, top + 15], [left, top + 15]])
            scores = [0.99] * len(boxes)
            scores[9] = 0.36  # unreadable amount text
            scores[12] = 0.41  # valid-looking but unreliable quantity
            return [{'rec_texts': [cell for row in rows for cell in row], 'rec_scores': scores, 'rec_polys': boxes}]

    class Vision:
        model = 'fake-vision'
        sizes = []

        def read_page(self, image, cancelled=None):
            with Image.open(io.BytesIO(image)) as bitmap:
                self.sizes.append(bitmap.size)
            if not html:
                return 'Product A | 1 ชิ้น | 18.69 | ไม่มีค่าบริการ\nProduct B | 1 | 10 | 10', 'stop'
            return (
                '<table><tr><th>No.</th><th>Description</th><th>จำนวน QUANTITY</th>'
                '<th>ราคาต่อหน่วย<br>UNIT PRICE</th><th>จำนวนเงิน AMOUNT</th></tr>'
                '<tr><td>1</td><td>Product A</td><td>1 ชิ้น</td><td>18.69</td><td>ไม่มีค่าบริการ</td></tr>'
                '<tr><td>2</td><td>Product B</td><td>1</td><td>10</td><td>10</td></tr></table>',
                'stop',
            )

    prompt = f'รอบนี้เมื่อพบ ไม่มีค่าบริการ ให้บันทึกยอด {amount} ตามกฎของงานนี้'
    vision = Vision()
    engine = OcrEngine(vlm=vision, llm=AnswerClient('http://x', 'm', 10, 8192, 1000, '-1', post=post))
    engine.vlm_mode = 'selective'
    engine.vlm_selective_min_confidence = 0.88
    engine.vlm_selective_min_chars = 0
    engine.pipeline, engine.vlm_ready, engine.llm_ready = Paddle(), True, True
    try:
        with fitz.open() as pdf:
            pdf.new_page(width=300, height=160)
            result = engine.process(pdf.tobytes(), 'application/pdf', prompt, response_schema=schema)
    finally:
        engine._executor.shutdown()

    assert len(vision.sizes) == 1
    width, height = vision.sizes[0]
    assert width == engine.vlm_max_long_side and height < width / 3
    assert len(requests) == (3 if focused else 2)
    assert all(request['messages'][1]['content'].startswith(f'คำสั่ง: {prompt}\n') for request in requests)
    assert 'อ3' in requests[0]['messages'][1]['content']
    assert 'อ3' not in requests[1]['messages'][1]['content']
    assert '<reread_table>' in requests[1]['messages'][1]['content']
    if focused:
        assert '<cells>' in requests[2]['messages'][1]['content']
    assert 'anyOf' in requests[0]['format']['properties']['items']['items']['properties']['amount']
    assert requests[0]['format']['properties']['items'].get('maxItems') == (None if corrected_name else 2)
    assert 'maxItems' not in schema['properties']['items']
    assert schema['properties']['items']['items']['properties']['amount'] == decimal
    assert result['pages'][0]['source'] == 'paddleocr'
    answer = result['answer']
    assert answer['error'] is None
    assert answer['content']['items'] == [
        {**first[0], 'quantity': None if 'quantity' in unresolved else '1',
         'amount': None if 'amount' in unresolved else amount,
         'net_amount': None if 'net_amount' in unresolved else amount},
        {**first[1], 'quantity': '1' if html else None},
    ]
    issues = {tuple(item['location']) for item in answer['review'] if item['status'] == 'invalid_decimal'}
    assert {location[2] for location in issues if location[:2] == ('items', 0)} == unresolved
    if 'amount' not in unresolved:
        assert answer['decimal_evidence']['$.items[0].amount'] == {
            'page_number': 1, 'row': 1, 'column': 'amount', 'reader': 'vlm',
            'method': 'interpreted', 'source_text': 'ไม่มีค่าบริการ', 'value': amount,
        }
