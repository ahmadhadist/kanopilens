"""Keep only the leaves of the plant being graded.

Nursery photos are full of other plants, and SAM's "leaf" prompt finds their
leaves too. A second SAM pass with "potted plant" outlines whole plants; the
plant that contains the subject leaf (large and near the centre of the frame)
is the one being graded. When that cannot be established, nothing is dropped:
grading a few background leaves is better than silently losing the plant's own.
"""

import math

import cv2
import numpy as np

MIN_OVERLAP = 0.5  # share of a leaf that must lie on the main plant
DILATE_RATIO = 0.04  # plant masks are a little tight around leaf tips


def _subject_score(mask: np.ndarray) -> float:
    h, w = mask.shape
    m = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    if m["m00"] == 0:
        return 0.0
    dist = math.hypot((m["m10"] / m["m00"] - w / 2) / w, (m["m01"] / m["m00"] - h / 2) / h)
    return (m["m00"] / (h * w)) * (1 - 0.7 * min(dist / 0.6, 1))


def main_plant(leaves: list[np.ndarray], plants: list[np.ndarray]) -> np.ndarray | None:
    """The (dilated) plant mask containing the subject leaf, or None if unclear."""
    if not leaves or not plants:
        return None
    anchor = max(leaves, key=_subject_score)
    best = max(plants, key=lambda p: int((p & anchor).sum()))
    if (best & anchor).sum() < MIN_OVERLAP * anchor.sum():
        return None
    r = max(3, int(DILATE_RATIO * max(best.shape)))
    return cv2.dilate(best.astype(np.uint8), np.ones((r, r), np.uint8)).astype(bool)


def on_plant(leaf: np.ndarray, plant: np.ndarray) -> bool:
    return (leaf & plant).sum() >= MIN_OVERLAP * leaf.sum()


def filter_leaves(leaves: list[np.ndarray], plants: list[np.ndarray]) -> list[np.ndarray]:
    plant = main_plant(leaves, plants)
    if plant is None:
        return leaves
    return [leaf for leaf in leaves if on_plant(leaf, plant)] or leaves


Tracks = dict[str, dict[int, np.ndarray]]


def filter_tracks(leaf_tracks: Tracks, plant_tracks: Tracks, frame_index: int) -> Tracks:
    """Pick the main plant on one reference frame, then keep the leaf tracks that sit
    on it in most of the frames where both are visible."""
    leaves_here = {oid: by[frame_index] for oid, by in leaf_tracks.items() if frame_index in by}
    plants_here = {pid: by[frame_index] for pid, by in plant_tracks.items() if frame_index in by}
    if not leaves_here or not plants_here:
        return leaf_tracks
    anchor = max(leaves_here.values(), key=_subject_score)
    main_id = max(plants_here, key=lambda pid: int((plants_here[pid] & anchor).sum()))
    if (plants_here[main_id] & anchor).sum() < MIN_OVERLAP * anchor.sum():
        return leaf_tracks

    r = max(3, int(DILATE_RATIO * max(anchor.shape)))
    kernel = np.ones((r, r), np.uint8)
    main = {i: cv2.dilate(m.astype(np.uint8), kernel).astype(bool) for i, m in plant_tracks[main_id].items()}
    kept = {}
    for oid, by_frame in leaf_tracks.items():
        shared = [i for i in by_frame if i in main]
        if not shared or sum(on_plant(by_frame[i], main[i]) for i in shared) * 2 >= len(shared):
            kept[oid] = by_frame
    return kept or leaf_tracks
