from app.field_labels import verify_field_labels


def line(text, x, y, width=95, confidence=0.99):
    return {'text': text, 'bbox': [[x, y], [x + width, y], [x + width, y + 10], [x, y + 10]],
            'confidence': confidence}


SPEC = {'properties': {'document': {'properties': {
    'tax_no': {'type': ['string', 'null']},
    'buyer_tax_id': {'type': ['string', 'null']},
}}}, 'x-ocr-field-labels': {
    'document.tax_no': {'labels': ['Tax Invoice No.']},
    'document.buyer_tax_id': {'labels': ['Buyer Tax ID'], 'scoped_labels': ['Tax ID'],
                              'sections': ['Buyer'], 'other_sections': ['Seller'], 'normalize_digits': True},
}}
BUYER = '0100000000001'
SELLER = '0100000000002'
INVOICE = '0000000000003'


def check(document, lines):
    answer = {'content': {'document': document}, 'review': []}
    verify_field_labels(answer, SPEC, [{'lines': lines, 'page_number': 1}])
    return answer


def test_corrects_swapped_identifiers_using_label_positions_even_when_both_have_thirteen_digits():
    result = check({'tax_no': BUYER, 'buyer_tax_id': INVOICE}, [
        line(f'Tax Invoice No. {INVOICE}', 0, 0),
        line('Seller: A Company', 0, 30),
        line('Buyer: B Company', 350, 30),
        line(f'Tax ID: {SELLER}', 0, 45),
        line('Tax ID:', 350, 45),
        line('0-1000-00000-00-1', 460, 45),
    ])
    assert result['content']['document'] == {'tax_no': INVOICE, 'buyer_tax_id': BUYER}
    assert {item['status'] for item in result['review']} == {'field_label_corrected'}
    assert result['field_evidence']['$.document.buyer_tax_id']['source_text'] == '0-1000-00000-00-1'


def test_drops_confirmed_wrong_field_without_guessing_missing_value():
    result = check({'tax_no': BUYER, 'buyer_tax_id': SELLER}, [
        line('Tax Invoice No.', 0, 0),
        line('Seller: A Company', 0, 35),
        line(f'Tax ID: {SELLER}', 0, 50),
        line('Buyer: B Company', 350, 35),
        line(f'Tax ID: {BUYER}', 350, 50),
    ])
    assert result['content']['document'] == {'tax_no': None, 'buyer_tax_id': BUYER}
    assert [item['status'] for item in result['review']] == ['field_label_wrong_field', 'field_label_corrected']


def test_unscoped_or_low_confidence_labels_are_flagged_for_review_without_moving_values():
    result = check({'tax_no': INVOICE, 'buyer_tax_id': '12345'}, [
        line(f'Tax Invoice No. {INVOICE}', 0, 0, confidence=0.5),
        line('Tax ID: 12345', 0, 40),
        line('Buyer: B Company', 350, 30),
    ])
    assert result['content']['document'] == {'tax_no': INVOICE, 'buyer_tax_id': '12345'}
    assert [item['status'] for item in result['review']] == ['field_label_unverified', 'field_label_unverified']


def test_optional_missing_values_are_preserved():
    result = check({'tax_no': None, 'buyer_tax_id': '12345'}, [
        line(f'Tax Invoice No. {INVOICE}', 0, 0),
        line('Buyer Tax ID: 12345', 0, 40),
    ])
    assert result['content']['document'] == {'tax_no': None, 'buyer_tax_id': '12345'}
    assert result['review'] == []


def test_scoped_invoice_number_and_offset_buyer_tax_id_with_a_shared_crop_reread():
    invoice_scope = {'scoped_labels': ['เลขที่', 'No.'], 'sections': ['Tax Invoice'],
                     'prefer_label': True, 'reread': True, 'shared_value': 'invoice_number'}
    fields = {
        'document.doc_no': {**invoice_scope, 'labels': ['Document No.']},
        'document.tax_no': {**invoice_scope, 'labels': ['Tax Invoice No.']},
        'document.buyer_tax_id': {'scoped_labels': ['Tax ID'], 'sections': ['SoldTo'], 'normalize_digits': True,
                                  'reject_before_section': True},
    }
    schema = {
        'properties': {'document': {'properties': {key.split('.')[-1]: {'type': ['string', 'null']} for key in fields}}},
        'x-ocr-field-labels': fields,
    }
    page = {'page_number': 1, 'lines': [
        line('Tax Invoice', 300, 20),
        line(f'Tax ID {SELLER}Branch Office', 0, 95, width=220),
        line('SoldTo', 0, 115),
        line('เลขที่', 275, 118, width=32),
        line('V690231', 312, 117),
        line('เลขที่ใบสั่งซื้อ999', 275, 135),
        line('P/O No. 98765', 275, 145),
        line('Tax ID', 0, 180, width=48),
        line('0 1000 00000 00 1', 50, 173),
    ]}
    answer = {'content': {'document': {'doc_no': 'V690231', 'tax_no': SELLER, 'buyer_tax_id': SELLER}}, 'review': []}
    crops = []

    def reread(page_number, box):
        crops.append((page_number, box))
        return 'IV690231'

    verify_field_labels(answer, schema, [page], reread)
    assert answer['content']['document'] == {'doc_no': 'IV690231', 'tax_no': 'IV690231', 'buyer_tax_id': BUYER}
    assert crops == [(1, (312, 117, 407, 127))]
    assert answer['field_evidence']['$.document.doc_no']['method'] == 'paddle_crop'

    distinct = {'content': {'document': {'doc_no': 'IV690231', 'tax_no': 'TAX-123', 'buyer_tax_id': SELLER}}, 'review': []}
    verify_field_labels(distinct, schema, [{**page, 'lines': [
        *page['lines'], line('Document No. DOC-123', 200, 195), line('Tax Invoice No. TAX-123', 200, 210),
    ]}])
    assert distinct['content']['document'] == {'doc_no': 'DOC-123', 'tax_no': 'TAX-123', 'buyer_tax_id': BUYER}
