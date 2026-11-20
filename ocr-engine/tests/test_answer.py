import json

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

    def answer(self, prompt, document, as_json):
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
    def post(path, payload, timeout):
        sent.update(payload)
        return {'message': {'content': '<think>x</think>{"a": 1}'}, 'done_reason': 'stop'}
    client = AnswerClient('http://x', 'm', 10, 8192, 1000, '-1', post=post)
    content, reason = client.answer('ช่วยเป็น json', 'เอกสาร', True)
    assert content == '{"a": 1}' and reason == 'stop'
    assert sent['format'] == 'json' and sent['options']['temperature'] == 0 and sent['options']['num_ctx'] == 8192
    assert '<document>\nเอกสาร\n</document>' in sent['messages'][1]['content']
    assert 'ห้ามเดา' in sent['messages'][0]['content']


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
