import json

import fitz
import pytest
from fastapi.testclient import TestClient

from app import main
from app.answer import AnswerClient, build_answer, check_answer, parse_json_answer, wants_json
from app.engine import OcrEngine

INVOICE = """บริษัท เบสท์ มูฟวิ่งแอนด์เซอร์วิส จำกัด
เลขประจำตัวผู้เสียภาษีอากร 0105550090936
เลขที่ IV690229 NO.
วันที่ 04/09/2569
<table><tr><td>2</td><td>ขายซาก OFS0010 ถ้าใบไม่ล่อน</td><td>1 ชิ้น</td><td>37.38</td><td>37.38</td></tr></table>
รวมจำนวนเงิน TOTAL AMOUNT 485.96
จำนวนเงินรวมทั้งสิ้น GRAND TOTAL 519.98"""

REVIEW = [{'text': 'OFS0010', 'reason': 'number_or_email_not_confirmed', 'similarity': None, 'page_number': 1}]


class FakeLlm:
    model = 'fake-llm'

    def __init__(self, content, reason='stop', error=None):
        self.content, self.reason, self.error = content, reason, error
        self.calls = []

    def answer(self, prompt, document, as_json, cancelled=None):
        self.calls.append((prompt, document, as_json))
        if self.error:
            raise self.error
        return self.content, self.reason


def test_wants_json_from_prompt_or_explicit_format():
    assert wants_json('ช่วยเป็น json', 'auto')
    assert not wants_json('สรุปให้หน่อย', 'auto')
    assert wants_json('สรุปให้หน่อย', 'json')
    assert not wants_json('ช่วยเป็น JSON', 'text')


def test_json_answer_values_are_checked_against_document():
    answer = {
        'invoice_no': 'IV690229',
        'tax_id': '0105550090936',
        'grand_total': 519.98,
        'subtotal': '485.96',
        'items': [{'code': 'OFS0010', 'description': 'ถ้าใบไม่ล่อน', 'amount': 37.38}],
        'customer_phone': '02-999-9999',  # invented: not in the document
        'po_number': None,
    }
    results = {r['path']: r for r in check_answer(answer, INVOICE, REVIEW)}
    assert results['$.customer_phone']['status'] == 'not_in_document'
    assert results['$.items[0].code']['status'] == 'uncertain_ocr'
    for ok in ['$.invoice_no', '$.tax_id', '$.grand_total', '$.subtotal', '$.items[0].amount', '$.po_number']:
        assert ok not in results, ok


def test_numbers_with_thousand_separators_match():
    assert check_answer({'total': '1,234.50'}, 'ยอด 1234.50', []) == []
    assert check_answer({'total': 1234.5}, 'ยอด 1,234.5', []) == []


def test_build_answer_returns_parsed_json_with_review():
    llm = FakeLlm(json.dumps({'invoice_no': 'IV690229', 'phone': '099-000-0000'}))
    result = build_answer(llm, True, None, 'ช่วยเป็น json', 'auto', INVOICE, REVIEW)
    assert result['format'] == 'json' and result['content']['invoice_no'] == 'IV690229'
    assert [r['path'] for r in result['review']] == ['$.phone']
    assert result['error'] is None and llm.calls[0][2] is True


def test_build_answer_reports_invalid_json_without_crashing():
    result = build_answer(FakeLlm('not json'), True, None, 'json please', 'auto', INVOICE, [])
    assert result['error'] == 'invalid_json' and result['content'] == 'not json'


def test_build_answer_text_mode_and_unavailable_llm():
    text = build_answer(FakeLlm('ยอดรวม 519.98 บาท'), True, None, 'ยอดรวมเท่าไร', 'auto', INVOICE, [])
    assert text['format'] == 'text' and text['content'] == 'ยอดรวม 519.98 บาท'
    down = build_answer(FakeLlm('x'), False, 'loading', 'json', 'auto', INVOICE, [])
    assert down['content'] is None and down['error'].startswith('llm_unavailable')
    failed = build_answer(FakeLlm('x', error=RuntimeError('boom')), True, None, 'json', 'auto', INVOICE, [])
    assert failed['error'].startswith('llm_failed')


def test_parse_json_answer_strips_code_fences():
    assert parse_json_answer('```json\n{"a": 1}\n```') == {'a': 1}


def test_chat_request_is_grounded_and_uses_json_format():
    sent = {}
    def post(path, payload, timeout, cancelled=None):
        sent.update(payload)
        return {'message': {'content': '<think>x</think>{"a": 1}'}, 'done_reason': 'stop'}
    client = AnswerClient('http://x', 'm', 10, 8192, 1000, '-1', post=post)
    content, reason = client.answer('ช่วยเป็น json', 'เอกสาร', True)
    assert content == '{"a": 1}' and reason == 'stop'
    assert sent['format'] == 'json' and sent['options']['temperature'] == 0 and sent['options']['num_ctx'] == 8192
    assert '<document>\nเอกสาร\n</document>' in sent['messages'][1]['content']
    assert 'ห้ามเดา' in sent['messages'][0]['content']


def test_structured_answer_preserves_ocr_rows_and_validates_items(monkeypatch):
    from tests.test_vlm_pipeline import png_bytes
    schema = {
        'type': 'object',
        'properties': {'items': {'type': 'array', 'items': {'type': 'object'}}},
        'required': ['items'],
        'additionalProperties': False,
    }
    received = []
    item = {'quantity': '1', 'price': '37000.00', 'amount': '37000.00'}
    answers = iter([{'items': [item]}, {'items': [item, 'withholding', []]}])

    def post(path, payload, timeout, cancelled=None):
        received.append(payload)
        return {'message': {'content': json.dumps(next(answers))}, 'done_reason': 'stop'}

    class InvoicePipeline:
        def predict(self, image):
            return [{
                'rec_texts': ['37,000.00', '1 งาน', 'GRAND TOTAL', '37,450.00'],
                'rec_polys': [
                    [[40, 18], [58, 18], [58, 25], [40, 25]],
                    [[15, 17], [30, 17], [30, 24], [15, 24]],
                    [[20, 45], [39, 45], [39, 54], [20, 54]],
                    [[40, 46], [58, 46], [58, 55], [40, 55]],
                ],
            }]

    monkeypatch.setenv('VLM_ENABLED', 'false')
    engine = OcrEngine(llm=AnswerClient('http://x', 'm', 10, 8192, 1000, '-1', post=post))
    engine.pipeline = InvoicePipeline()
    engine.llm_ready = True
    engine.upscale_min_long_side = 0
    monkeypatch.setattr(engine, 'load', lambda: None)
    monkeypatch.setattr(main, 'engine', engine)

    with TestClient(main.app) as client:
        def submit(raw_schema):
            return client.post('/ocr', data={'prompt': 'Read the items', 'response_schema': raw_schema},
                               files={'file': ('invoice.png', png_bytes(), 'image/png')})

        assert submit('{').status_code == 422
        assert submit(json.dumps({**schema, 'required': True})).status_code == 422
        assert received == []
        result = submit(json.dumps(schema)).json()['answer']
        assert result['content'] == {'items': [item]} and result['error'] is None
        assert received[0]['format'] == schema
        user_prompt = received[0]['messages'][1]['content']
        assert '1 งาน | 37,000.00\nGRAND TOTAL | 37,450.00' in user_prompt
        assert '"sources"' not in user_prompt

        invalid = submit(json.dumps(schema)).json()['answer']
        assert invalid['content'] == {'items': [item]} and invalid['error'] is None
        assert invalid['review'][0]['status'] == 'response_schema_mismatch'


@pytest.mark.parametrize(('first_price', 'second_price', 'second_rows', 'expected_price', 'retries'), [
    ('8อ3', '803', 2, '803', 1),
    ('8อ3', 'C', 2, None, 1),
    ('8อ3', '803', 1, None, 1),
    ('8อ3', '830', 2, None, 1),
    ('8อ3', '90', 2, None, 1),
    ('803', None, 2, '803', 0),
    (None, '803', 2, None, 0),
    ('', '803', 2, None, 0),
    (' \t\n', '803', 2, None, 0),
])
def test_selective_pdf_retries_only_invalid_decimal_fields(first_price, second_price, second_rows, expected_price, retries):
    decimal = {'type': ['string', 'null'], 'pattern': r'^[0-9]+(?:\.[0-9]+)?$', 'x-ocr-decimal': True}
    schema = {
        'type': 'object', 'required': ['items'], 'properties': {'items': {'type': 'array', 'items': {
            'type': 'object', 'required': ['name', 'price', 'amount'],
            'properties': {'name': {'type': 'string'}, 'price': decimal, 'amount': decimal},
            'additionalProperties': False,
        }}},
    }
    first = [{'name': 'Product A', 'price': first_price, 'amount': '600'}, {'name': 'Product B', 'price': '20', 'amount': '20'}]
    second = [
        {'name': 'Product A', 'price': second_price, 'amount': '900'},
        {'name': 'Product B', 'price': '90', 'amount': '90'},
    ][:second_rows]
    responses = iter([first, second])
    requests = []

    def post(path, payload, timeout, cancelled=None):
        requests.append(payload)
        return {'message': {'content': json.dumps({'items': next(responses)})}, 'done_reason': 'stop'}

    class Paddle:
        def predict(self, image):
            return [{'rec_texts': ['Invoice line ' * 12, 'Product A 600', 'Product B 20', '8อ3'],
                     'rec_scores': [0.99, 0.98, 0.98, 0.71]}]

    class Vision:
        model = 'fake-vision'
        calls = 0

        def read_page(self, image, cancelled=None):
            self.calls += 1
            return 'Product A | 803 | 900\nProduct B | 90', 'stop'

    vision = Vision()
    engine = OcrEngine(vlm=vision, llm=AnswerClient('http://x', 'm', 10, 8192, 1000, '-1', post=post))
    engine.vlm_mode = 'selective'
    engine.vlm_selective_min_confidence = 0.88
    engine.pipeline, engine.vlm_ready, engine.llm_ready = Paddle(), True, True
    try:
        with fitz.open() as pdf:
            pdf.new_page(width=100, height=100)
            result = engine.process(pdf.tobytes(), 'application/pdf', 'Read the items', response_schema=schema)
    finally:
        engine._executor.shutdown()

    assert vision.calls == retries and len(requests) == 1 + retries
    assert requests[0]['format']['properties']['items']['items']['properties']['price'] == {'type': ['string', 'null']}
    assert schema['properties']['items']['items']['properties']['price'] == decimal and 'pattern' in decimal
    assert result['pages'][0]['source'] == 'paddleocr'
    assert result['answer']['error'] is None
    assert result['answer']['content']['items'] == [{**first[0], 'price': expected_price}, first[1]]
    bad = [r for r in result['answer']['review'] if r['status'] == 'invalid_decimal']
    if retries and expected_price is None:
        assert bad == [{'path': '$.items[0].price', 'location': ['items', 0, 'price'], 'value': None, 'status': 'invalid_decimal'}]
    else:
        assert bad == []


def test_engine_adds_answer_when_prompt_given():
    from tests.test_vlm_pipeline import FakePipeline, FakeVlm, png_bytes
    llm = FakeLlm(json.dumps({'contact': 'คุณเจนจิรา'}))
    engine = OcrEngine(vlm=FakeVlm('คุณเจนจิรา 098-765-4310'), llm=llm)
    engine.pipeline = FakePipeline()
    engine.vlm_ready = engine.llm_ready = True
    result = engine.process(png_bytes(), 'image/png', 'ช่วยเป็น json', 'auto')
    assert result['answer']['content'] == {'contact': 'คุณเจนจิรา'}
    assert result['answer']['review'] == []
    assert result['engines']['llm'] == 'fake-llm'
    no_prompt = engine.process(png_bytes(), 'image/png')
    assert 'answer' not in no_prompt


# Header of the real Typhoon output for the BEST MOVING tax invoice; the P/O line is empty.
INVOICE_HEADER = """นามผู้ซื้อ คุณณัฐธยาน์ พวงคำ SOLD TO
บ้านเลขที่ 133 ถนนเชียงใหม่
ที่อยู่ ADDRESSแขวงคลองสาน เขตคลองสาน กรุงเทพมหานคร 10600

เลขประจำตัวผู้เสียภาษีอากร 3 1018 00812 46 2

เลขที่ IV690229 NO.
วันที่ 04/09/2569 DATE
เงื่อนไขการชำระเงิน cash TERMS OF PAYMENT
เลขที่ใบสั่งซื้อ P/O NO.

<table><tr><td>1</td><td>ขายซาก AS5912-0003</td><td>1 ชิ้น</td></tr><tr><td>2</td><td>ขายซาก OFS0010</td><td>37.38</td></tr></table>
จำนวนเงินรวมทั้งสิ้น GRAND TOTAL 519.98"""


def test_value_taken_from_another_label_is_flagged():
    from app.answer import check_sources
    data = {'invoice_no': 'IV690229', 'buyer_phone': None, 'po_number': '3 1018 00812 46 2', 'grand_total': '519.98'}
    sources = {'invoice_no': 'เลขที่', 'buyer_phone': None, 'po_number': 'เลขที่ใบสั่งซื้อ P/O NO.',
               'grand_total': 'จำนวนเงินรวมทั้งสิ้น GRAND TOTAL'}
    results = {r['path']: r for r in check_sources(data, sources, INVOICE_HEADER)}
    assert results['$.po_number']['status'] == 'label_mismatch'
    assert set(results) == {'$.po_number'}


def test_invented_label_and_missing_label_are_flagged():
    from app.answer import check_sources
    data = {'invoice_no': 'IV690229', 'buyer_tax_id': '3 1018 00812 46 2'}
    results = {r['path']: r['status'] for r in check_sources(data, {'invoice_no': 'Order reference code'}, INVOICE_HEADER)}
    assert results == {'$.invoice_no': 'label_not_found', '$.buyer_tax_id': 'label_missing'}


def test_build_answer_unwraps_data_and_checks_sources():
    content = json.dumps({'data': {'invoice_no': 'IV690229', 'po_number': '3 1018 00812 46 2'},
                          'sources': {'invoice_no': 'เลขที่', 'po_number': 'เลขที่ใบสั่งซื้อ P/O NO.'}}, ensure_ascii=False)
    result = build_answer(FakeLlm(content), True, None, 'ขอข้อมูลเป็น json: invoice_no, po_number', 'auto', INVOICE_HEADER, [])
    assert result['content'] == {'invoice_no': 'IV690229', 'po_number': '3 1018 00812 46 2'}
    assert result['sources']['po_number'] == 'เลขที่ใบสั่งซื้อ P/O NO.'
    assert [(r['path'], r['status']) for r in result['review']] == [('$.po_number', 'label_mismatch')]


def test_build_answer_accepts_plain_json_without_sources_wrapper():
    result = build_answer(FakeLlm('{"invoice_no": "IV690229"}'), True, None, 'json', 'auto', INVOICE_HEADER, [])
    assert result['content'] == {'invoice_no': 'IV690229'} and result['sources'] is None and result['review'] == []


def test_system_prompt_forbids_borrowing_values_from_other_labels():
    from app.answer import JSON_INSTRUCTION, SYSTEM_PROMPT
    assert 'ห้ามนำค่าที่อยู่ใกล้ๆ ของป้ายอื่นมาใส่แทน' in SYSTEM_PROMPT
    assert '"sources"' in JSON_INSTRUCTION
