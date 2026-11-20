import time

import numpy as np

from app.crosscheck import document_words, find_missing, find_suspicious
from app.engine import OcrEngine
from app.vlm import VlmClient, parse_keep_alive, prepare_vlm_image

# PaddleOCR output for the sample receipt (upscaled run), trimmed to the relevant lines.
PADDLE = """ซึ่อลุกค้า
บริษัก ยืมเอเจนธีแอนด์ อิจิกอล จากัด
559/21 อนนพหลโยริน แขวงจอนพล เขตจตุจักร กรุดกพษ 10900
เลบผู้เสียภาธิ
098-765-4310
helloobeardigiral.co.th
ผูติดซ่อ
คุณเจนจิรา
เบอร์โกรศัพท์
098-765-4310
88/8 หนู่ที่ 3 ต่านสบางแน่นาง จาเกอ
บางใหญ่จังหวัอนนกบุรี11140
9,630 บาก"""

# Typhoon output with the errors it actually produced on the sample.
TYPHOON = """ชื่อลูกค้า บริษัท บีบีเอเจนซี่ แอนด์ ดีวิกอล จำกัด
ที่อยู่ 559/21 ถนนพหลโยธิน แขวงจอมพล เขตจตุจักร
เลขผู้เสียภาษี 098-765-4310 อีเมล์ hello@boeandigital.co.th
ผู้ติดต่อ คุณเจนจิรา 058-765-4310
ที่อยู่ 88/8 หมู่ที่ 3 ตำบลบางบอนป่าขาว อำเภอบางใหญ่ จังหวัดสมุทรปราการ 11140
<figure>
ภาพประกอบเป็นรูปแมวลายเสือสีส้ม
</figure>
<table><tr><td>จำนวนเงินรวมทั้งสิ้น</td><td>**9,630** บาท</td></tr></table>"""


def flagged(markdown: str) -> set:
    return {item['text'] for item in find_suspicious(markdown, PADDLE)}


def test_invented_and_misread_values_are_flagged():
    flags = flagged(TYPHOON)
    for wrong in ['058-765-4310', 'hello@boeandigital.co.th', 'ตำบลบางบอนป่าขาว', 'จังหวัดสมุทรปราการ', 'บีบีเอเจนซี่']:
        assert wrong in flags, wrong


def test_correct_values_supported_by_paddle_are_not_flagged():
    flags = flagged(TYPHOON)
    for right in ['098-765-4310', 'คุณเจนจิรา', 'ถนนพหลโยธิน', 'แขวงจอมพล', '11140', '9,630']:
        assert right not in flags, right


def test_figure_descriptions_and_markup_are_ignored():
    words = document_words(TYPHOON)
    assert 'ภาพประกอบเป็นรูปแมวลายเสือสีส้ม' not in words
    assert '9,630' in words and 'td' not in words


def test_crosscheck_is_fast_enough():
    started = time.time()
    find_suspicious(TYPHOON * 5, PADDLE * 5)
    assert time.time() - started < 2.0


def test_prepare_vlm_image_enlarges_small_and_shrinks_large():
    from PIL import Image
    import io
    png, factor = prepare_vlm_image(np.zeros((660, 465, 3), dtype=np.uint8), 1300, 1800)
    assert round(factor, 3) == round(1300 / 660, 3)
    assert Image.open(io.BytesIO(png)).size == (round(465 * factor), 1300)
    _, factor = prepare_vlm_image(np.zeros((3600, 2400, 3), dtype=np.uint8), 1300, 1800)
    assert factor == 0.5


class FakePipeline:
    def predict(self, image):
        return [{'rec_texts': ['คุณเจนจิรา', '098-765-4310'], 'rec_scores': [0.9, 0.99],
                 'rec_polys': [[[0, 0], [10, 0], [10, 10], [0, 10]]] * 2}]


class FakeVlm:
    model = 'fake-typhoon'

    def __init__(self, output=None, error=None):
        self.output, self.error = output, error

    def read_page(self, png):
        if self.error:
            raise self.error
        return self.output, 'stop'


def make_engine(vlm, ready=True):
    engine = OcrEngine(vlm=vlm)
    engine.pipeline = FakePipeline()
    engine.vlm_ready = ready
    engine.vlm_wait_seconds = 0
    return engine


def png_bytes():
    from PIL import Image
    import io
    buffer = io.BytesIO()
    Image.new('RGB', (100, 80), 'white').save(buffer, format='PNG')
    return buffer.getvalue()


def test_engine_uses_vlm_text_and_reports_review():
    result = make_engine(FakeVlm('คุณเจนจิรา 058-765-4310')).process(png_bytes(), 'image/png')
    page = result['pages'][0]
    assert page['source'] == 'vlm' and page['text'].startswith('คุณเจนจิรา')
    assert page['paddle_text'] == 'คุณเจนจิรา\n098-765-4310'
    assert [r['text'] for r in result['review']] == ['058-765-4310']
    assert result['review'][0]['page_number'] == 1
    assert result['engines']['vlm'] == 'fake-typhoon'


def test_engine_falls_back_to_paddle_and_flags_when_vlm_fails():
    result = make_engine(FakeVlm(error=RuntimeError('boom'))).process(png_bytes(), 'image/png')
    assert result['pages'][0]['source'] == 'paddleocr'
    assert result['review'][0]['reason'] == 'vlm_failed'


def test_engine_flags_when_vlm_not_ready():
    result = make_engine(FakeVlm('x'), ready=False).process(png_bytes(), 'image/png')
    assert result['pages'][0]['source'] == 'paddleocr'
    assert result['review'][0]['reason'] == 'vlm_unavailable'
    assert result['engines']['vlm'] is None


def test_keep_alive_is_sent_as_number_when_numeric():
    assert parse_keep_alive('-1') == -1
    assert parse_keep_alive(' 300 ') == 300
    assert parse_keep_alive('30m') == '30m'
    assert VlmClient('http://x', 'm', 1, 1, '-1').keep_alive == -1


def test_vlm_load_retries_until_ollama_is_ready():
    class Flaky:
        model = 'm'
        calls = 0
        def ensure_model(self):
            Flaky.calls += 1
            if Flaky.calls < 3:
                raise RuntimeError('not yet')
    engine = OcrEngine(vlm=Flaky())
    engine._load_vlm(attempts=5, wait_seconds=0)
    assert engine.vlm_ready and engine.vlm_error is None and Flaky.calls == 3


def test_vlm_load_reports_last_error_after_giving_up():
    class Broken:
        model = 'm'
        def ensure_model(self):
            raise RuntimeError('down')
    engine = OcrEngine(vlm=Broken())
    engine._load_vlm(attempts=2, wait_seconds=0)
    assert not engine.vlm_ready and 'down' in engine.vlm_error and '2/2' in engine.vlm_error


def test_lines_skipped_by_the_vlm_are_reported():
    # Real run: Typhoon dropped the issuer address; PaddleOCR saw it (noisily).
    vlm = open(__file__.replace('test_vlm_pipeline.py', 'data/typhoon_dropped_address.md'), encoding='utf-8').read()
    paddle_lines = [
        {'text': 'บริษัก ยืมเอเจนธีแอนด์ อิจิกอล จากัด', 'confidence': 0.91, 'bbox': None},
        {'text': 'ฮ้พงซัง', 'confidence': 0.65, 'bbox': None},
        {'text': '88/8 หนู่ที่ 3 ต่านสบางแน่นาง จาเกอ', 'confidence': 0.86, 'bbox': [[75, 186]]},
        {'text': 'บางใหญ่จังหวัอนนกบุรี11140', 'confidence': 0.87, 'bbox': None},
        {'text': 'บัตรo0ิ0/ นัดรครอัก', 'confidence': 0.74, 'bbox': None},
        {'text': ', รันสาระ', 'confidence': 0.72, 'bbox': None},
        {'text': 'ร', 'confidence': 0.4, 'bbox': None},
    ]
    missing = find_missing(vlm, paddle_lines)
    assert [m['text'] for m in missing] == ['88/8 หนู่ที่ 3 ต่านสบางแน่นาง จาเกอ', 'บางใหญ่จังหวัอนนกบุรี11140']
    assert missing[0]['bbox'] == [[75, 186]] and missing[0]['reason'] == 'missing_in_vlm'


def test_job_waits_for_vlm_that_is_still_loading():
    import threading
    engine = make_engine(FakeVlm('คุณเจนจิรา 098-765-4310'), ready=False)
    engine.vlm_wait_seconds = 5
    def finish_loading():
        time.sleep(0.2)
        engine.vlm_ready = True
        engine._vlm_loaded.set()
    threading.Thread(target=finish_loading).start()
    started = time.time()
    result = engine.process(png_bytes(), 'image/png')
    assert result['pages'][0]['source'] == 'vlm'
    assert 0.1 < time.time() - started < 3


def test_product_codes_are_checked_by_digits_not_letters():
    paddle = 'ขายซาก OR50016 ทาโพลีน\nขายชาก 0F50024สว่าน\nขายซาก OP50026 บล้อคลม\nชายซาก OF50010 ผ้าใบ\nVICTOR/VY-2221'
    vlm = 'ขายซาก OF50016 ขายซาก OF50024 ขายซาก OF50026 ขายซาก OFS0010 VICTOR#VY-222n'
    flags = {f['text'] for f in find_suspicious(vlm, paddle)}
    assert 'OF50016' not in flags and 'OF50024' not in flags and 'OF50026' not in flags
    assert 'OFS0010' in flags  # digits 0010 vs 50010: a real misread
    assert 'VY-222n' in flags  # digits differ from PaddleOCR (2221); ambiguous, so checked by hand


def test_vlm_request_sets_context_window_and_unescapes_html():
    import json as _json
    sent = {}
    client = VlmClient('http://x', 'm', 1, 100, '-1', num_ctx=12288)
    def fake_post(path, payload, timeout):
        sent.update(payload)
        return {'response': 'BEST MOVING&amp;SERVICE CO.,LTD.', 'done_reason': 'stop'}
    client._post = fake_post
    text, reason = client.read_page(b'png')
    assert sent['options']['num_ctx'] == 12288 and sent['options']['num_predict'] == 100
    assert text == 'BEST MOVING&SERVICE CO.,LTD.' and reason == 'stop'
