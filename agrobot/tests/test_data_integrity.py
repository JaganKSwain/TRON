"""Integrity tests for the data layer.

These assert the properties the headline metric depends on. If any fails, the
reported accuracy is not trustworthy, so they are written to be independent of
the code that produced the splits: they re-derive leakage from the manifest
columns rather than trusting `verify_splits`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agrobot.data.manifest import (
    MIN_DUP_CORR,
    THUMBS_NPY,
    _near_duplicate_pairs,
    confirm_pairs,
    dhash_to_hex,
    discover_class_dirs,
    hashes_from_hex,
    load_manifest,
    parse_filename,
)
from agrobot.paths import DATA_ROOT, MANIFEST_CSV, NON_CLASS_DIRS, SPLITS_CSV

pytestmark = pytest.mark.skipif(
    not SPLITS_CSV.exists(), reason="run agrobot.data.manifest and agrobot.data.splits first"
)

HELD_OUT = ("test", "val", "test_site")


@pytest.fixture(scope="module")
def splits() -> pd.DataFrame:
    return load_manifest(SPLITS_CSV)


@pytest.fixture(scope="module")
def manifest() -> pd.DataFrame:
    return load_manifest(MANIFEST_CSV)


# ---------------------------------------------------------------------------
# Filename parsing -- the grouping key depends entirely on this
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("name", "site", "leaf_core", "shot", "day"),
    [
        ("00fc2ee5-x___RS_HL 1864.JPG", "RS_HL", "RS_HL 1864", "", ""),
        ("000146ff-x___GH_HL Leaf 259.1.JPG", "GH_HL", "GH_HL Leaf 259", "1", ""),
        ("d0badc95-x___GHLB Leaf 10 Day 6.jpeg", "GHLB", "GHLB Leaf 10", "", "6"),
        ("x___GHLB Leaf 2.1 Day 16.JPG", "GHLB", "GHLB Leaf 2", "1", "16"),
        ("svn-r6Yb5c.JPG", "svn-r6Yb5c", "svn-r6Yb5c", "", ""),
    ],
)
def test_parse_filename(name, site, leaf_core, shot, day):
    p = parse_filename(name)
    assert (p.site, p.leaf_core, p.shot, p.day) == (site, leaf_core, shot, day)


def test_shot_and_day_variants_share_a_leaf_core():
    """`Leaf 2.1 Day 16` and `Leaf 2 Day 3` are one physical leaf."""
    a = parse_filename("u___GHLB Leaf 2.1 Day 16.JPG")
    b = parse_filename("v___GHLB Leaf 2 Day 3.JPG")
    assert a.leaf_core == b.leaf_core


# ---------------------------------------------------------------------------
# Manifest hygiene
# ---------------------------------------------------------------------------
def test_nested_duplicate_directory_is_excluded():
    """`<DATA_ROOT>/PlantVillage/` mirrors every class; scanning it doubles the data."""
    assert "PlantVillage" in NON_CLASS_DIRS
    names = [d.name for d in discover_class_dirs(DATA_ROOT)]
    assert "PlantVillage" not in names
    assert len(names) == 15


def test_stray_tool_directories_are_not_treated_as_classes(tmp_path):
    """A tool cache in the dataset root must not become a 16th class.

    Found the hard way: running pytest from the dataset root creates
    `.pytest_cache`, whose `.gitignore` matched an `or p.suffix == ""` clause in
    the image test, so the directory looked like a class. Class names are sorted
    to build the label mapping, and `.pytest_cache` sorts *first* -- so every
    label index would have shifted by one. Silent, and catastrophic for training.
    """
    real = tmp_path / "Tomato_healthy"
    real.mkdir()
    (real / "a.jpg").write_bytes(b"\xff\xd8\xff\xe0stub")

    for junk, payload in [(".pytest_cache", ".gitignore"),
                          (".ipynb_checkpoints", "notes"),
                          ("__pycache__", "mod.cpython-311.pyc")]:
        d = tmp_path / junk
        d.mkdir()
        (d / payload).write_text("x")

    empty = tmp_path / "Not_A_Class"  # holds only a zero-byte non-image
    empty.mkdir()
    (empty / "svn-r6Yb5c").write_bytes(b"")

    names = [d.name for d in discover_class_dirs(tmp_path)]
    assert names == ["Tomato_healthy"], f"stray directories leaked in: {names}"


def test_dhash_survives_the_csv_round_trip(manifest):
    """Read as int this column exceeds 2**53 and float64 silently rounds it."""
    # Assert the property, not the dtype: pandas 2 gives object, pandas 3 StringDtype.
    assert not pd.api.types.is_numeric_dtype(manifest["dhash"])
    assert manifest["dhash"].map(type).eq(str).all()
    h = hashes_from_hex(manifest["dhash"])
    assert [dhash_to_hex(int(v)) for v in h[:50]] == list(manifest["dhash"][:50])
    # Any collision here would mean precision was lost somewhere.
    assert len(np.unique(h)) == len(h)


def test_no_exact_duplicates_remain(manifest):
    assert not manifest["md5"].duplicated().any()


def test_no_label_conflicts_remain(manifest):
    """The same leaf must not carry two disease labels."""
    if "label_conflict" in manifest:
        assert not manifest["label_conflict"].any()
    spanning = manifest.groupby("group_id")["class"].nunique()
    assert int((spanning > 1).sum()) == 0


# ---------------------------------------------------------------------------
# Split leakage -- re-derived from scratch, not read back from verify_splits
# ---------------------------------------------------------------------------
def test_all_images_assigned_exactly_once(splits):
    assert splits["split"].isin(["train", "val", "test", "test_site"]).all()
    assert not splits["path"].duplicated().any()


def test_leaf_identity_never_spans_two_splits(splits):
    """The core guarantee: Day 6 in train and Day 9 in test would inflate the score."""
    spanning = splits.groupby(["class", "leaf_core"])["split"].nunique()
    offenders = spanning[spanning > 1]
    assert offenders.empty, f"{len(offenders)} leaf identities span splits: {list(offenders.index[:5])}"


def test_group_never_spans_two_splits(splits):
    spanning = splits.groupby("group_id")["split"].nunique()
    assert spanning.max() == 1


def test_md5_never_spans_two_splits(splits):
    spanning = splits.groupby("md5")["split"].nunique()
    assert spanning.max() == 1


def test_no_confirmed_near_duplicate_spans_a_split(splits):
    """Independent leakage check: rebuild near-duplicates and locate both ends."""
    if not THUMBS_NPY.exists():
        pytest.skip("thumbs.npy not built")
    thumbs = np.load(THUMBS_NPY)
    assert len(thumbs) == len(splits), "thumbnails are not row-aligned with the manifest"

    candidates, mih = _near_duplicate_pairs(hashes_from_hex(splits["dhash"]), 4)
    assert mih["exhaustive"], "candidate search was not exhaustive"
    confirmed, _ = confirm_pairs(candidates, thumbs, MIN_DUP_CORR)

    spl = splits["split"].to_numpy()
    straddling = [(a, b) for a, b in confirmed if spl[a] != spl[b]]
    assert not straddling, f"{len(straddling)} confirmed near-duplicates straddle splits"


def test_every_class_present_in_train_val_test(splits):
    for name in ("train", "val", "test"):
        present = set(splits.loc[splits["split"] == name, "class"])
        assert present == set(splits["class"]), f"'{name}' is missing classes"


def test_held_out_sites_do_not_leak_into_training(splits):
    """A site withheld for cross-site evaluation must appear nowhere else."""
    n_checked = 0
    for cls, sub in splits.groupby("class"):
        held = set(sub.loc[sub["split"] == "test_site", "site"])
        if not held:
            continue
        n_checked += 1
        trainable = set(sub.loc[sub["split"].isin(["train", "val"]), "site"])
        assert not (held & trainable), f"{cls}: {held & trainable} leaked into train/val"
    assert n_checked == 6, f"expected 6 multi-site classes with a holdout, got {n_checked}"


def test_split_proportions_are_close_to_target(splits):
    """Groups are indivisible, so fractions are approximate -- but not wildly off."""
    trainable = splits[splits["split"] != "test_site"]
    shares = trainable["split"].value_counts(normalize=True)
    assert shares["train"] == pytest.approx(0.70, abs=0.03)
    assert shares["val"] == pytest.approx(0.15, abs=0.03)
    assert shares["test"] == pytest.approx(0.15, abs=0.03)


# ---------------------------------------------------------------------------
# Near-duplicate machinery
# ---------------------------------------------------------------------------
def test_multi_index_hashing_matches_brute_force():
    """The pigeonhole shortcut must not miss pairs a full scan would find."""
    rng = np.random.default_rng(0)
    base = rng.integers(0, 2**63, size=300, dtype=np.uint64)
    # Seed in some genuine near-duplicates by flipping a few bits.
    variants = base[:20] ^ np.array([1 << i for i in range(20)], dtype=np.uint64)
    h = np.concatenate([base, variants])

    found, stats = _near_duplicate_pairs(h, 4)
    assert stats["exhaustive"]

    xor = h[:, None] ^ h[None, :]
    dist = np.bitwise_count(xor) if hasattr(np, "bitwise_count") else None
    if dist is None:
        pytest.skip("numpy < 2.0")
    iu = np.triu_indices(len(h), k=1)
    expected = {(int(a), int(b)) for a, b in zip(*[i[dist[iu] <= 4] for i in iu])}
    assert set(found) == expected


def test_confirm_pairs_rejects_uncorrelated_thumbnails():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 255, size=(32, 32), dtype=np.uint8)
    thumbs = np.stack([a, a.copy(), rng.integers(0, 255, size=(32, 32), dtype=np.uint8)])
    kept, corrs = confirm_pairs([(0, 1), (0, 2)], thumbs, MIN_DUP_CORR)
    assert kept == [(0, 1)]
    assert corrs[0] == pytest.approx(1.0, abs=1e-5)
    assert corrs[1] < MIN_DUP_CORR
