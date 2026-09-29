"""Tests for leaf segmentation and mask QC (P2).

The QC flag decides whether an image gets `RandomBackgroundSwap`, so a
systematically wrong flag would quietly withhold the augmentation that exists to
break the capture-site shortcut. These tests pin the properties that matter, on
synthetic images where ground truth is known exactly.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from agrobot.data.segment import (
    MAX_AREA_FRAC,
    MAX_LEAK_FRAC,
    MIN_AREA_FRAC,
    MIN_LARGEST_CC_FRAC,
    _fill_holes,
    segment_leaf,
)

SIZE = 256


def _synth(backdrop: tuple[int, int, int], leaf: tuple[int, int, int],
           axes: tuple[int, int] = (70, 95), noise: float = 3.0) -> np.ndarray:
    """A leaf-like ellipse on a uniform backdrop, in BGR."""
    img = np.zeros((SIZE, SIZE, 3), np.uint8)
    img[:] = backdrop
    cv2.ellipse(img, (SIZE // 2, SIZE // 2), axes, 20, 0, 360, leaf, -1)
    rng = np.random.default_rng(0)
    img = np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return img


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0, b > 0
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 0.0


def _truth(axes: tuple[int, int] = (70, 95)) -> np.ndarray:
    t = np.zeros((SIZE, SIZE), np.uint8)
    cv2.ellipse(t, (SIZE // 2, SIZE // 2), axes, 20, 0, 360, 255, -1)
    return t


# ---------------------------------------------------------------------------
# Does it segment at all?
# ---------------------------------------------------------------------------
def test_segments_a_green_leaf_on_grey():
    mask, stats = segment_leaf(_synth((190, 190, 190), (60, 140, 70)))
    assert stats.ok, stats.reason
    assert _iou(mask, _truth()) > 0.95


def test_mask_polarity_is_leaf_not_backdrop():
    """255 must mean leaf. An inverted mask would sail through area checks."""
    mask, _ = segment_leaf(_synth((190, 190, 190), (60, 140, 70)))
    centre = mask[SIZE // 2, SIZE // 2]
    corner = mask[2, 2]
    assert centre == 255 and corner == 0


def test_segments_a_necrotic_brown_leaf():
    """The reason for LAB-distance over an excess-green index: diseased leaves
    are brown, and ExG would score them as background."""
    mask, stats = segment_leaf(_synth((190, 190, 190), (40, 75, 120)))
    assert stats.ok, stats.reason
    assert _iou(mask, _truth()) > 0.95


# ---------------------------------------------------------------------------
# The regression test for the bug this rule replaced
# ---------------------------------------------------------------------------
def test_tinted_backdrop_does_not_trip_qc():
    """`Tomato_Leaf_Mold` (site `Crnl_L.Mold`) is shot on a dark blue-grey card.

    The previous rule compared mean chroma inside the mask against outside, on
    the assumption that a leaf is more colourful than a grey backdrop. This
    site's backdrop is dark *and* saturated -- BGR (91, 56, 46), measured from
    the dataset -- so in LAB it is more chromatic than the dark green leaf and
    the comparison inverts. It flagged 117 of that class's images, whose masks
    were in fact the cleanest in the dataset (leak_frac max 0.036).

    That mattered because a failed flag disables `RandomBackgroundSwap`: the one
    class defined by an unusual backdrop would have been the one class denied the
    augmentation meant to sever the capture-site/label correlation.

    The colours below are the real ones, so this test genuinely reproduces the
    old failure -- `chroma_inside <= chroma_outside` holds here.
    """
    img = _synth((91, 56, 46), (51, 75, 40), noise=2.0)
    mask, stats = segment_leaf(img)
    assert stats.chroma_inside <= stats.chroma_outside, (
        "this case no longer reproduces the old chroma inversion; the test has "
        "stopped guarding the bug it was written for"
    )
    assert stats.ok, f"tinted backdrop wrongly flagged: {stats.reason}"
    assert _iou(mask, _truth()) > 0.95


def test_desaturated_leaf_does_not_trip_qc():
    """Pale `Tomato_healthy` leaves are barely more chromatic than a grey card."""
    img = _synth((186, 186, 186), (120, 132, 122))
    mask, stats = segment_leaf(img)
    assert stats.ok, f"pale leaf wrongly flagged: {stats.reason}"
    assert _iou(mask, _truth()) > 0.90


def test_real_leaf_mold_images_pass_qc():
    """The 117 formerly-flagged images, checked against the real dataset."""
    import pandas as pd

    from agrobot.paths import REPORTS_DIR

    qc_csv = REPORTS_DIR / "mask_qc.csv"
    if not qc_csv.exists():
        pytest.skip("run data/segment.py first")
    qc = pd.read_csv(qc_csv)
    cls = qc["path"].str.replace("\\", "/", regex=False).str.split("/").str[-2]
    lm = qc[cls == "Tomato_Leaf_Mold"]
    if lm.empty:
        pytest.skip("Tomato_Leaf_Mold not present")

    # The old rule fired on these; the new one must not.
    inverted_chroma = lm[lm["chroma_inside"] <= lm["chroma_outside"]]
    assert len(inverted_chroma) > 0, "expected this site to invert the chroma test"
    assert inverted_chroma["ok"].all(), (
        f"{(~inverted_chroma['ok']).sum()} tinted-backdrop images still flagged"
    )


# ---------------------------------------------------------------------------
# Does QC actually fire? A silent check is worse than none.
# ---------------------------------------------------------------------------
def test_qc_rejects_an_inverted_mask():
    """leak_frac must catch inversion regardless of backdrop hue.

    Calibrated on 400 real images: correct masks p99 0.22, inverted min 0.51.
    """
    from agrobot.data.segment import _mask_from_distance, _qc

    img = _synth((190, 190, 190), (60, 140, 70))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    frame = np.ones((SIZE, SIZE), bool)
    frame[6:-6, 6:-6] = False
    bg = np.median(lab[frame], axis=0)
    dist = np.sqrt((((lab - bg) * np.array([0.5, 1.5, 1.5], np.float32)) ** 2).sum(axis=2))
    good, cc, leafish = _mask_from_distance(dist, img, False)

    assert _qc(good, img, lab, frame, cc, leafish, "border").ok
    bad = _qc(255 - good, img, lab, frame, cc, leafish, "border")
    assert not bad.ok and "leak_frac" in bad.reason
    assert bad.leak_frac > MAX_LEAK_FRAC


def test_qc_rejects_a_partial_mask():
    """Masks covering only part of a leaf are the genuine `Target_Spot` failure."""
    from agrobot.data.segment import _mask_from_distance, _qc

    img = _synth((190, 190, 190), (60, 140, 70))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    frame = np.ones((SIZE, SIZE), bool)
    frame[6:-6, 6:-6] = False
    bg = np.median(lab[frame], axis=0)
    dist = np.sqrt((((lab - bg) * np.array([0.5, 1.5, 1.5], np.float32)) ** 2).sum(axis=2))
    full, cc, leafish = _mask_from_distance(dist, img, False)

    partial = full.copy()
    partial[: SIZE // 2, :] = 0  # keep only the lower half of the leaf
    stats = _qc(partial, img, lab, frame, cc, leafish, "border")
    assert not stats.ok and "leak_frac" in stats.reason


def test_qc_rejects_an_empty_and_a_full_mask():
    """No leaf found, and everything-is-leaf, are both failures."""
    img = _synth((190, 190, 190), (60, 140, 70))
    _, stats = segment_leaf(img)
    assert MIN_AREA_FRAC < stats.area_frac < MAX_AREA_FRAC
    assert 0.0 <= stats.leak_frac <= 1.0
    assert stats.largest_cc_frac >= MIN_LARGEST_CC_FRAC


def test_qc_thresholds_are_ordered_sanely():
    assert 0.0 < MIN_AREA_FRAC < MAX_AREA_FRAC < 1.0
    assert 0.0 < MIN_LARGEST_CC_FRAC <= 1.0
    # Must sit above the p99 of correct masks (0.22) and below the minimum
    # observed on inverted masks (0.51), or the rule stops discriminating.
    assert 0.22 < MAX_LEAK_FRAC < 0.51


# ---------------------------------------------------------------------------
# Hole filling: lesion spots read as backdrop and must not punch through
# ---------------------------------------------------------------------------
def test_fill_holes_closes_interior_gaps():
    m = np.zeros((100, 100), np.uint8)
    cv2.rectangle(m, (20, 20), (80, 80), 255, -1)
    cv2.circle(m, (50, 50), 8, 0, -1)  # a lesion spot
    filled = _fill_holes(m)
    assert filled[50, 50] == 255
    assert filled[5, 5] == 0, "flood fill must not leak into the exterior"


def test_lesion_spots_do_not_punch_holes_in_the_mask():
    """A diseased leaf is still one solid region."""
    img = _synth((190, 190, 190), (60, 140, 70))
    rng = np.random.default_rng(1)
    for _ in range(25):  # scatter dark lesions across the leaf
        c = (int(rng.integers(200, 312)) - 128 + SIZE // 2 - 56,
             int(rng.integers(200, 312)) - 128 + SIZE // 2 - 56)
        cv2.circle(img, c, int(rng.integers(3, 7)), (35, 45, 70), -1)
    mask, stats = segment_leaf(img)
    assert stats.ok, stats.reason
    assert _iou(mask, _truth()) > 0.90


# ---------------------------------------------------------------------------
# The dataset-level gate from the plan
# ---------------------------------------------------------------------------
def test_dataset_mask_qc_pass_rate_meets_the_gate():
    """P2's gate: >=95% of masks pass QC. Skips until `segment.py` has run."""
    import pandas as pd

    from agrobot.paths import REPORTS_DIR

    qc_csv = REPORTS_DIR / "mask_qc.csv"
    if not qc_csv.exists():
        pytest.skip("run data/segment.py first")
    qc = pd.read_csv(qc_csv)
    rate = qc["ok"].mean()
    assert rate >= 0.95, f"mask QC pass rate {rate:.2%} below the 95% gate"

    # No single class may dominate the failures: that pattern means a broken QC
    # *rule*, and it would strip background-swap augmentation from exactly one
    # class -- the failure mode that motivated leak_frac.
    bad = qc[~qc["ok"]]
    if len(bad) >= 10:
        cls = bad["path"].str.replace("\\", "/", regex=False).str.split("/").str[-2]
        top = cls.value_counts(normalize=True).iloc[0]
        assert top < 0.60, (
            f"{top:.0%} of mask QC failures are one class "
            f"({cls.value_counts().index[0]}) -- suspect the rule, not the masks"
        )
