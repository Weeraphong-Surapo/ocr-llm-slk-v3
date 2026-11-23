import numpy as np
from app.ocr_pipeline import normalize_result, paddle_layout_text, scale_bbox, sniff_file_type, upscale_small_image
def test_small_image_is_enlarged_up_to_target_long_side():
    image = np.zeros((660, 465, 3), dtype=np.uint8)
    out, factor = upscale_small_image(image, 2000, 4.0)
    assert round(factor, 3) == round(2000 / 660, 3)
    assert out.shape[:2] == (2000, round(465 * factor))
def test_factor_is_capped():
    image = np.zeros((100, 50, 3), dtype=np.uint8)
    out, factor = upscale_small_image(image, 2000, 4.0)
    assert factor == 4.0
    assert out.shape[:2] == (400, 200)
def test_large_image_and_disabled_setting_are_unchanged():
    large = np.zeros((2500, 1800, 3), dtype=np.uint8)
    assert upscale_small_image(large, 2000, 4.0)[1] == 1.0
    small = np.zeros((100, 100, 3), dtype=np.uint8)
    out, factor = upscale_small_image(small, 0, 4.0)
    assert factor == 1.0 and out is small
def test_bbox_is_mapped_back_to_original_pixels():
    assert scale_bbox([[81, 60], [615, 90], [612, 201], [81, 195]], 3.0) == [[27, 20], [205, 30], [204, 67], [27, 65]]
    assert scale_bbox(None, 3.0) is None
def test_normalize_result_converts_numpy_boxes():
    result = {'rec_texts': ['ทดสอบ'], 'rec_scores': [0.9], 'rec_polys': [np.array([[1, 2], [3, 2], [3, 4], [1, 4]])]}
    lines = normalize_result(result)['lines']
    assert lines[0]['bbox'] == [[1, 2], [3, 2], [3, 4], [1, 4]]


def test_sniff_file_type_uses_content_not_declared_type():
    assert sniff_file_type(b'%PDF-1.7\n...') == 'application/pdf'
    assert sniff_file_type(b'\x89PNG\r\n\x1a\n....') == 'image/png'
    assert sniff_file_type(b'\xff\xd8\xff\xe0....') == 'image/jpeg'
    assert sniff_file_type(b'RIFF\x00\x00\x00\x00WEBPVP8 ') == 'image/webp'
    assert sniff_file_type(b'hello') is None


def _cell(text, left, right, y):
    return {'text': text, 'bbox': [[left, y - 10], [right, y - 10], [right, y + 10], [left, y + 10]]}


def test_parallel_shifted_columns_preserve_the_amounts_for_each_label():
    labels = ['SUBTOTAL', 'DISCOUNT', 'TAX BASE', 'VAT', 'GRAND TOTAL']
    amounts = ['485.96', '0.00', '485.96', '34.02', '519.98']
    lines = [_cell(label, 730, 950, 1119 + 34 * row) for row, label in enumerate(labels)]
    lines += [_cell(amount, 1050, 1125, 1098 + 34 * row) for row, amount in enumerate(amounts)]
    lines.append(_cell('7.00%', 890, 950, 1200))

    output = paddle_layout_text(lines).splitlines()

    assert [line for line in output if ' | ' in line] == [f'{label} | {amount}' for label, amount in zip(labels, amounts)]
    assert '7.00%' in output


def test_a_missing_amount_does_not_shift_other_rows():
    labels = ['FIRST', 'SECOND', 'THIRD', 'FOURTH', 'FIFTH']
    lines = [_cell(label, 730, 950, 100 + 34 * row) for row, label in enumerate(labels)]
    lines += [_cell(f'{row}.00', 1050, 1125, 100 + 34 * row) for row in [0, 1, 3, 4]]

    assert paddle_layout_text(lines).splitlines() == ['FIRST | 0.00', 'SECOND | 1.00', 'THIRD', 'FOURTH | 3.00', 'FIFTH | 4.00']
