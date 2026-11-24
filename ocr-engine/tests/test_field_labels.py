from copy import deepcopy

import pytest

from app.field_labels import _restores_prefix, verify_field_labels


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


@pytest.mark.parametrize('missing', [None, '', '   '])
def test_opted_in_missing_tax_number_uses_only_a_unique_label_value(missing):
    schema = deepcopy(SPEC)
    schema['x-ocr-field-labels']['document.tax_no']['fill_missing'] = True
    tax_number = line(f'Tax Invoice No. {INVOICE}', 0, 0)
    for lines, expected in [
        ([tax_number], INVOICE),
        ([tax_number, line('Tax Invoice No. IV2000', 0, 20)], missing),
        ([line('P/O No. 123456', 0, 0)], missing),
    ]:
        answer = {'content': {'document': {'tax_no': missing, 'buyer_tax_id': None}}, 'review': []}
        verify_field_labels(answer, schema, [{'page_number': 1, 'lines': lines}])
        assert answer['content']['document'] == {'tax_no': expected, 'buyer_tax_id': None}
        if expected == INVOICE:
            assert answer['review'] == [{'path': '$.document.tax_no', 'value': missing, 'status': 'field_label_filled'}]
            assert answer['field_evidence']['$.document.tax_no']['source_text'] == INVOICE
        else:
            assert answer['review'] == []


@pytest.mark.parametrize('first_pass', ['V690231', 'I690231'])
def test_scoped_invoice_number_and_offset_buyer_tax_id_with_a_shared_crop_reread(first_pass):
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
        line(first_pass, 312, 117),
        line('เลขที่ใบสั่งซื้อ999', 275, 135),
        line('P/O No. 98765', 275, 145),
        line('Tax ID', 0, 180, width=48),
        line('0 1000 00000 00 1', 50, 173),
    ]}
    answer = {'content': {'document': {'doc_no': first_pass, 'tax_no': SELLER, 'buyer_tax_id': SELLER}}, 'review': []}
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


@pytest.mark.parametrize(('first_pass', 'crop'), [
    ('I690232', 'IV690233'),
    ('X690232', 'IV690232'),
    ('VI690232', 'IVX690232'),
    ('IV-690232', 'IVX690232'),
])
def test_crop_cannot_overwrite_digits_separators_or_existing_prefix_letters(first_pass, crop):
    assert not _restores_prefix(first_pass, crop)
