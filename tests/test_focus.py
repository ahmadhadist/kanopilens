import numpy as np

from app.services import focus, variegation


def box(y0, y1, x0, x1, shape=(100, 100)):
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


SUBJECT = box(30, 70, 35, 65)  # big leaf in the centre: the plant being graded
SIBLING = box(20, 40, 60, 75)  # another leaf of the same plant
BACKGROUND = box(0, 15, 0, 15)  # leaf of a plant behind it
MAIN_PLANT = box(15, 80, 30, 80)
OTHER_PLANT = box(0, 20, 0, 20)


def test_background_leaves_are_dropped():
    kept = focus.filter_leaves([BACKGROUND, SUBJECT, SIBLING], [OTHER_PLANT, MAIN_PLANT])
    assert [k.sum() for k in kept] == [SUBJECT.sum(), SIBLING.sum()]


def test_nothing_dropped_when_main_plant_is_unclear():
    # Seen on real photos: SAM's "potted plant" sometimes outlines only the pot.
    pot_only = box(80, 100, 30, 70)
    leaves = [BACKGROUND, SUBJECT, SIBLING]
    assert focus.filter_leaves(leaves, [pot_only, OTHER_PLANT]) is leaves
    assert focus.filter_leaves(leaves, []) is leaves


def test_tracks_are_filtered_on_the_reference_frame():
    leaf_tracks = {"1": {0: SUBJECT, 1: SUBJECT}, "2": {0: SIBLING}, "3": {0: BACKGROUND, 1: BACKGROUND}, "4": {5: SIBLING}}
    plant_tracks = {"a": {0: MAIN_PLANT, 1: MAIN_PLANT}, "b": {0: OTHER_PLANT}}
    kept = focus.filter_tracks(leaf_tracks, plant_tracks, frame_index=0)
    # "4" never appears together with the main plant, so there is no evidence to drop it.
    assert sorted(kept) == ["1", "2", "4"]


def test_pink_counts_only_for_pink_cultivars():
    pink = np.full((1, 4), 170, np.uint8), np.full((1, 4), 120, np.uint8), np.full((1, 4), 200, np.uint8)
    base = variegation.DEFAULT_THRESHOLDS
    for hint, expected in [("Philodendron Pink Princess", 4), ("Alocasia Strawberry Shake", 4), ("Colocasia", 0), (None, 0)]:
        variegated, _ = variegation.classify_hsv(*pink, variegation.for_species(base, hint))
        assert variegated.sum() == expected, hint
