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
