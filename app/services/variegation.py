"""Deterministic variegation measurement inside leaf masks.

Variegation is computed per pixel from colour, not guessed by an LLM, so the
same photo always yields the same number.
"""

import json
from dataclasses import dataclass, replace
from functools import lru_cache

import cv2
import numpy as np

from app.schemas import LeafResult, VariegationGrade, VariegationSummary

GRADES: tuple[VariegationGrade, ...] = ("none", "low", "medium", "high", "very_high")


@dataclass(frozen=True)
class Thresholds:
    """OpenCV HSV ranges (H 0-179, S/V 0-255) and grade limits.

    Defaults are generic; production values calibrated against expert-graded
    photos are supplied through the VARIEGATION_THRESHOLDS setting.
    """

    white_max_s: int = 60
    white_min_v: int = 150
    yellow_min_h: int = 18
    yellow_max_h: int = 34
    yellow_min_s: int = 70
    yellow_min_v: int = 130
    pink_low_max_h: int = 8
    pink_high_min_h: int = 155
    pink_min_s: int = 40
    pink_min_v: int = 150  # high enough to keep reddish-brown necrosis out of "pink"
    dark_max_v: int = 40  # deep shadows are excluded from both numerator and denominator
    # Blown-out pixels (glare on glossy leaves) carry no colour, so they are excluded
    # like shadows. 256 disables this; calibration decides whether it helps.
    blown_min_v: int = 256
    # Upper bound (exclusive, % of leaf area) of none, low, medium, high; above is very_high.
    grade_limits: tuple[float, float, float, float] = (2, 15, 35, 60)


DEFAULT_THRESHOLDS = Thresholds()


@lru_cache
def thresholds_from_json(text: str) -> Thresholds:
    if not text:
        return DEFAULT_THRESHOLDS
    overrides = json.loads(text)
    if "grade_limits" in overrides:
        overrides["grade_limits"] = tuple(overrides["grade_limits"])
    return replace(DEFAULT_THRESHOLDS, **overrides)


def classify_hsv(h: np.ndarray, s: np.ndarray, v: np.ndarray, t: Thresholds) -> tuple[np.ndarray, np.ndarray]:
    """Return (variegated, valid) for OpenCV HSV channel arrays of any shape."""
    white = (s < t.white_max_s) & (v > t.white_min_v)
    yellow = (h >= t.yellow_min_h) & (h <= t.yellow_max_h) & (s >= t.yellow_min_s) & (v >= t.yellow_min_v)
    pink = ((h <= t.pink_low_max_h) | (h >= t.pink_high_min_h)) & (s >= t.pink_min_s) & (v >= t.pink_min_v)
    valid = (v > t.dark_max_v) & (v < t.blown_min_v)
    return (white | yellow | pink) & valid, valid


# Pink counts as variegation only on cultivars where pink sectors are the
# variegation; elsewhere red or pink leaves (Colocasia, Calathea) are base colour.
PINK_VARIEGATED = ("pink", "princess", "ppp", "strawberry", "tricolor", "tri-color")


def for_species(t: Thresholds, species_hint: str | None) -> Thresholds:
    hint = (species_hint or "").lower()
    if any(word in hint for word in PINK_VARIEGATED):
        return t
    return replace(t, pink_min_s=256)  # saturation never reaches 256: pink disabled


def variegation_mask(rgb: np.ndarray, t: Thresholds = DEFAULT_THRESHOLDS) -> tuple[np.ndarray, np.ndarray]:
    """Return (variegated, valid) boolean masks over the whole image."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return classify_hsv(hsv[..., 0], hsv[..., 1], hsv[..., 2], t)


def measure_leaves(
    rgb: np.ndarray, masks: list[np.ndarray], t: Thresholds = DEFAULT_THRESHOLDS
) -> list[LeafResult]:
    """Measure each mask; LeafResult.index is the mask's position in `masks`."""
    variegated, valid = variegation_mask(rgb, t)
    results = []
    for i, mask in enumerate(masks):
        counted = int((mask & valid).sum())
        if counted == 0:
            continue
        pct = 100.0 * int((mask & variegated).sum()) / counted
        ys, xs = np.nonzero(mask)
        x, y = int(xs.min()), int(ys.min())
        bbox = (x, y, int(xs.max()) - x + 1, int(ys.max()) - y + 1)
        results.append(LeafResult(index=i, bbox=bbox, area_px=int(mask.sum()), variegation_pct=round(pct, 1)))
    return results


def grade_for(pct: float, t: Thresholds = DEFAULT_THRESHOLDS) -> VariegationGrade:
    for limit, grade in zip(t.grade_limits, GRADES):
        if pct < limit:
            return grade
    return "very_high"


def summarize(leaves: list[LeafResult], t: Thresholds = DEFAULT_THRESHOLDS) -> VariegationSummary | None:
    if not leaves:
        return None
    pcts = [leaf.variegation_pct for leaf in leaves]
    areas = [leaf.area_px for leaf in leaves]
    weighted = sum(p * a for p, a in zip(pcts, areas)) / sum(areas)
    return VariegationSummary(
        weighted_pct=round(weighted, 1),
        mean_pct=round(sum(pcts) / len(pcts), 1),
        min_pct=min(pcts),
        max_pct=max(pcts),
        grade=grade_for(weighted, t),
    )
