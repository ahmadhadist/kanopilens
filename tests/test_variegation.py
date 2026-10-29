import numpy as np

from app.services.variegation import grade_for, measure_leaves, summarize, thresholds_from_json

GREEN = (40, 120, 50)
WHITE = (240, 240, 230)
YELLOW = (230, 200, 40)
SHADOW = (10, 15, 10)


def leaf_image(left: tuple, right: tuple) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.full((100, 100, 3), (120, 80, 50), np.uint8)  # brown background
    rgb[20:80, 20:50] = left
    rgb[20:80, 50:80] = right
    mask = np.zeros((100, 100), bool)
    mask[20:80, 20:80] = True
    return rgb, mask


def test_half_white_leaf_is_50_percent():
    rgb, mask = leaf_image(GREEN, WHITE)
    [leaf] = measure_leaves(rgb, [mask])
    assert leaf.variegation_pct == 50.0
    assert leaf.bbox == (20, 20, 60, 60)


def test_yellow_counts_as_variegation():
    rgb, mask = leaf_image(GREEN, YELLOW)
    [leaf] = measure_leaves(rgb, [mask])
    assert leaf.variegation_pct == 50.0


def test_all_green_leaf_is_zero():
    rgb, mask = leaf_image(GREEN, GREEN)
    [leaf] = measure_leaves(rgb, [mask])
    assert leaf.variegation_pct == 0.0


def test_shadow_is_excluded_from_denominator():
    rgb, mask = leaf_image(WHITE, SHADOW)
    [leaf] = measure_leaves(rgb, [mask])
    assert leaf.variegation_pct == 100.0


def test_summary_is_area_weighted():
    rgb, big = leaf_image(GREEN, GREEN)
    rgb[0:10, 0:10] = WHITE
    small = np.zeros((100, 100), bool)
    small[0:10, 0:10] = True
    leaves = measure_leaves(rgb, [big, small])
    summary = summarize(leaves)
    assert summary.mean_pct == 50.0
    assert summary.weighted_pct == round(100 * 100 / (3600 + 100), 1)
    assert summary.grade == "low"


def test_grades():
    assert [grade_for(p) for p in (0, 10, 20, 50, 80)] == ["none", "low", "medium", "high", "very_high"]


def test_thresholds_override_from_json():
    t = thresholds_from_json('{"grade_limits": [1, 5, 10, 20], "white_min_v": 250}')
    assert grade_for(12, t) == "high"
    rgb, mask = leaf_image(GREEN, WHITE)  # WHITE has V=240, now below the white threshold
    [leaf] = measure_leaves(rgb, [mask], t)
    assert leaf.variegation_pct == 0.0
