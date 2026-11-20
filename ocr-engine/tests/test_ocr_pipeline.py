import numpy as np
from app.ocr_pipeline import normalize_result, scale_bbox, upscale_small_image
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
