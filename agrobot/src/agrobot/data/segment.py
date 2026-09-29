"""Leaf segmentation, cached to disk.

Masks are needed for three things, and the third is the one that matters most:

1. `RandomBackgroundSwap` -- cut the leaf out and paste it on a new background.
2. Harvesting background pixels to synthesise the `__not_a_leaf__` reject class.
3. The **background-only ablation** in `diagnose.py`: blank the leaf, keep only
   the backdrop, and check the model collapses to chance. Without masks there
   is no way to prove the model is not reading the capture session.

Method: PlantVillage backdrops are uniform (studio grey, occasionally dark or
tinted) while leaves are not. Rather than an excess-green index -- which fails
badly on necrotic brown/yellow leaves, i.e. exactly the diseased ones we care
about -- we estimate the backdrop colour from a border frame and threshold the
per-pixel LAB distance from it. That adapts to whatever the backdrop happens to
be and is indifferent to leaf hue. Leaves that fill the frame defeat the border
reference, so a chroma-only fallback covers them; see `segment_leaf`.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agrobot.data.manifest import load_manifest  # noqa: E402
from agrobot.paths import (  # noqa: E402
    MASK_CACHE,
    REPORTS_DIR,
    SPLITS_CSV,
    ensure_dirs,
)

# Weights on the LAB axes when measuring distance from the backdrop. Chroma is
# weighted above luminance: shadows change L a lot without indicating leaf,
# whereas a/b separate foliage from a grey card. Shared with `residue_frac` so
# the QC check and the segmenter agree on what "unlike the backdrop" means.
LAB_WEIGHTS = np.array([0.5, 1.5, 1.5], np.float32)

# QC thresholds. A mask outside these bounds is flagged, not silently trusted.
MIN_AREA_FRAC = 0.08
MAX_AREA_FRAC = 0.97
MIN_LARGEST_CC_FRAC = 0.80
# Share of backdrop-unlike ("leafish") pixels the mask leaves outside itself.
# Calibrated against synthetically inverted masks over a 400-image sample:
# correct masks p99 = 0.22, max 0.49; inverted masks min 0.51, p01 0.78. Any
# threshold in 0.25-0.50 catches 100% of inversions; 0.30 was chosen because it
# also catches partial masks (see `_qc`) at a ~1% false-alarm rate.
MAX_LEAK_FRAC = 0.30

# Opening kernel. 3, not 5: a 5x5 open erodes petioles, leaf tips and serrated
# margins away entirely, and those pixels then survive into the background bank
# and into the P5 ablation. Measured over 600 images, dropping to 3 cut leaf
# tissue left outside the generous mask by 3x (mean 0.0056 -> 0.0018 of frame,
# worst case 0.272 -> 0.126).
OPEN_PX = 3
# Components smaller than these are noise; anything larger is kept, because a
# compound tomato leaf can legitimately segment into several blobs. Rarely fires
# (2 images in 600) but costs nothing and is the correct rule.
MIN_KEEP_REL = 0.10
MIN_KEEP_FRAC = 0.005

# `generous_mask` parameters. Used by background harvesting, the `__not_a_leaf__`
# class and the P5 background-only ablation -- every consumer that must be sure no
# leaf pixel survives, and for which over-removing backdrop is harmless.
# Closing first, then dilating: a 13px dilation cannot span a 40px void, and
# growing the silhouette by 40px to reach one would swallow half the frame.
CLOSE_PX = 41
DILATE_PX = 13

# `residue_frac` parameters, calibrated over the 600 harvest sources.
RESID_LAB_THRESH = 20.0    # weighted LAB distance clearly beyond sensor noise
RESID_TEX_MIN = 300.0      # Laplacian variance: tissue 700-20k, cast shadow 15-65
RESID_MIN_AREA = 0.002     # ignore specks below 0.2% of frame


@dataclass
class MaskStats:
    area_frac: float
    largest_cc_frac: float
    border_frac: float
    leak_frac: float
    resid_frac: float
    chroma_inside: float
    chroma_outside: float
    ok: bool
    reason: str
    method: str = "border"


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes (lesion spots often read as background)."""
    h, w = mask.shape
    ff = mask.copy()
    pad = np.zeros((h + 2, w + 2), np.uint8)
    pad[1:-1, 1:-1] = ff
    # Flood from a known-outside pixel; whatever stays 0 is an interior hole.
    cv2.floodFill(pad, None, (0, 0), 255)
    holes = (pad[1:-1, 1:-1] == 0).astype(np.uint8) * 255
    return cv2.bitwise_or(mask, holes)


def generous_mask(
    mask: np.ndarray, close_px: int = CLOSE_PX, dilate_px: int = DILATE_PX
) -> np.ndarray:
    """Grow a mask until no leaf pixel can plausibly sit outside it.

    For the three consumers that must not leave leaf tissue behind -- background
    harvesting, the `__not_a_leaf__` reject class, and the P5 background-only
    ablation -- a mask that is slightly too big costs nothing, while one that is
    slightly too small invalidates the result. So: close, fill, then dilate.

    The order matters. Masks that pass QC still contain notches: washed-out or
    specular leaf regions that read as backdrop and connect to the mask boundary,
    which `_fill_holes` cannot close by construction (flood fill reaches them from
    outside). Closing spans them without moving the outer silhouette; dilation
    cannot, because reaching a 40px notch would mean growing 40px outward in every
    direction as well.
    """
    if close_px:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
        mask = _fill_holes(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k))
    if dilate_px:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        mask = cv2.dilate(mask, k)
    return mask


def residue_frac(bgr: np.ndarray, mask: np.ndarray, border: int = 6) -> float:
    """Largest patch of *leaf tissue* left outside the generous mask, as a share
    of the frame.

    This is the QC statistic that `leak_frac` cannot be. `leak_frac` compares the
    mask against the Otsu map, and both come from the same LAB distance map -- so
    when the distance map itself misses part of the leaf (bright veins, specular
    midribs, thin petioles), mask and map agree and the check reads clean. This
    one is referenced only to the backdrop and to image texture, so it sees what
    the distance map lost.

    Texture is the discriminator, not colour: a leaf's cast shadow is also unlike
    the backdrop, but it *belongs* in a harvested backdrop. Measured over 600
    images, surviving tissue runs 700-20,000 Laplacian variance while shadows and
    vignette run 15-65, so `RESID_TEX_MIN` separates them with room to spare.
    """
    outside = generous_mask(mask) == 0
    if not outside.any():
        return 0.0

    h, w = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    frame = np.ones((h, w), bool)
    frame[border:-border, border:-border] = False
    # Reference from border pixels that are outside the mask, hence known backdrop.
    ref_px = lab[frame & outside]
    ref = np.median(ref_px if len(ref_px) >= 50 else lab[outside], axis=0)

    dist = np.sqrt((((lab - ref) * LAB_WEIGHTS) ** 2).sum(axis=2))
    hot = ((dist > RESID_LAB_THRESH) & outside).astype(np.uint8)
    hot = cv2.morphologyEx(
        hot, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    n, labels, cc_stats, _ = cv2.connectedComponentsWithStats(hot, connectivity=8)
    if n <= 1:
        return 0.0

    lap = cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_32F)
    frame_px = float(h * w)
    worst = 0.0
    for i in range(1, n):
        area = cc_stats[i, cv2.CC_STAT_AREA] / frame_px
        if area <= max(worst, RESID_MIN_AREA):
            continue
        if float(lap[labels == i].var()) >= RESID_TEX_MIN:
            worst = area
    return worst


def _mask_from_distance(
    dist: np.ndarray, bgr: np.ndarray, use_grabcut: bool
) -> tuple[np.ndarray, float, np.ndarray]:
    """Otsu-threshold a distance map, then clean it into one solid blob.

    Also returns the raw pre-morphology threshold map: it is the algorithm's own
    estimate of "unlike the backdrop", and comparing the finished mask against it
    is what `_qc` uses to tell a good mask from a partial one.
    """
    dist_u8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    leafish = mask > 0

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_PX, OPEN_PX))
    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k9, iterations=2)

    # Keep every component that is a meaningful share of the biggest. Keeping only
    # the largest discards the second leaflet of a compound tomato leaf, and those
    # pixels then survive into the background bank and the P5 ablation.
    n_lab, labels, stats_cc, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    largest_cc_frac = 1.0
    if n_lab > 1:
        areas = stats_cc[1:, cv2.CC_STAT_AREA]
        total = float(areas.sum())
        largest_cc_frac = float(areas.max() / total) if total > 0 else 0.0
        floor = max(MIN_KEEP_REL * areas.max(), MIN_KEEP_FRAC * mask.size)
        keep = [i + 1 for i, a in enumerate(areas) if a >= floor]
        if not keep:  # every component is tiny; fall back to the biggest
            keep = [int(np.argmax(areas)) + 1]
        mask = np.where(np.isin(labels, keep), 255, 0).astype(np.uint8)

    mask = _fill_holes(mask)

    if use_grabcut and mask.any():
        mask = _grabcut_refine(bgr, mask)
        mask = _fill_holes(mask)
    return mask, largest_cc_frac, leafish


def _qc(
    mask: np.ndarray,
    bgr: np.ndarray,
    lab: np.ndarray,
    frame: np.ndarray,
    largest_cc_frac: float,
    leafish: np.ndarray,
    method: str,
) -> MaskStats:
    """Flag masks that cannot be trusted for background swapping or ablation.

    The decisive check is `leak_frac`: of the pixels that look nothing like the
    estimated backdrop, what share did the mask fail to include? It catches the
    two failures that matter with one number, and -- crucially -- it is
    **referenced to the backdrop this image actually has**, so it works on the
    tinted backdrops and washed-out leaves that defeat an absolute colour test:

    * **Inverted** (mask holds the backdrop): every leafish pixel is outside, so
      leak -> 1.0.
    * **Partial** (mask holds only the darkest patches of a pale leaf): most of
      the leaf is outside, so leak is large.

    A previous version compared mean chroma inside against outside, on the
    assumption that a leaf is more colourful than a grey card. That assumption is
    false for `Tomato_Leaf_Mold` (shot on a blue-grey backdrop) and for pale
    `Tomato_healthy` leaves, and it produced 139 false alarms -- 117 from that
    one class -- while passing genuinely broken `Tomato__Target_Spot` masks.
    Chroma is still recorded below, but only as a diagnostic.

    `resid_frac` is recorded but deliberately **not** folded into `ok`. It answers
    a different question -- "did any leaf tissue escape the generous mask?" -- and
    the consumers want different thresholds for it. `RandomBackgroundSwap` happily
    tolerates a small fragment left behind; background harvesting, the
    `__not_a_leaf__` class and the P5 ablation do not. Collapsing two distinct
    questions into one boolean is precisely what made the chroma rule harmful.
    """
    area_frac = float(mask.mean() / 255.0)
    border_frac = float((mask[frame] > 0).mean())
    chroma = np.sqrt(((lab[:, :, 1] - 128) ** 2) + ((lab[:, :, 2] - 128) ** 2))
    inside = mask > 0
    chroma_in = float(chroma[inside].mean()) if inside.any() else 0.0
    chroma_out = float(chroma[~inside].mean()) if (~inside).any() else 0.0

    n_leafish = int(leafish.sum())
    leak_frac = float((leafish & ~inside).sum() / n_leafish) if n_leafish else 1.0

    reason = ""
    if area_frac < MIN_AREA_FRAC:
        reason = f"area_frac {area_frac:.3f} < {MIN_AREA_FRAC}"
    elif area_frac > MAX_AREA_FRAC:
        reason = f"area_frac {area_frac:.3f} > {MAX_AREA_FRAC}"
    elif largest_cc_frac < MIN_LARGEST_CC_FRAC:
        reason = f"fragmented: largest_cc_frac {largest_cc_frac:.3f}"
    elif leak_frac > MAX_LEAK_FRAC:
        reason = f"leak_frac {leak_frac:.3f} > {MAX_LEAK_FRAC} (inverted or partial)"

    return MaskStats(
        area_frac=area_frac,
        largest_cc_frac=largest_cc_frac,
        border_frac=border_frac,
        leak_frac=leak_frac,
        resid_frac=residue_frac(bgr, mask),
        chroma_inside=chroma_in,
        chroma_outside=chroma_out,
        ok=not reason,
        reason=reason,
        method=method,
    )


def segment_leaf(
    bgr: np.ndarray, border: int = 6, use_grabcut: bool = False
) -> tuple[np.ndarray, MaskStats]:
    """Return (uint8 mask 0/255, stats). 255 = leaf.

    Two strategies, tried in order:

    1. **border** -- estimate the backdrop colour from a border frame and
       threshold LAB distance from it. Adapts to whatever the backdrop happens
       to be (studio grey, dark, tinted) and is indifferent to leaf hue.
    2. **chroma** -- fallback for leaves that fill the frame. When the leaf
       touches the border, the reference is sampled *off the leaf itself*, the
       distance map inverts and the mask shatters. Thresholding absolute chroma
       instead needs no reference: PlantVillage backdrops are near-neutral grey
       (a,b close to 128) while foliage -- green, yellow or necrotic brown --
       is not. It is the weaker cue in general, which is why it is second.
    """
    h, w = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    frame = np.ones((h, w), bool)
    frame[border:-border, border:-border] = False
    bg_ref = np.median(lab[frame], axis=0)  # (L, a, b)

    # Distance from backdrop, chroma weighted above luminance (see LAB_WEIGHTS).
    dist = np.sqrt((((lab - bg_ref) * LAB_WEIGHTS) ** 2).sum(axis=2))
    mask, cc, leafish = _mask_from_distance(dist, bgr, use_grabcut)
    stats = _qc(mask, bgr, lab, frame, cc, leafish, "border")
    if stats.ok:
        return mask, stats

    chroma_dist = np.sqrt((lab[:, :, 1] - 128.0) ** 2 + (lab[:, :, 2] - 128.0) ** 2)
    alt_mask, alt_cc, alt_leafish = _mask_from_distance(chroma_dist, bgr, use_grabcut)
    alt_stats = _qc(alt_mask, bgr, lab, frame, alt_cc, alt_leafish, "chroma")
    if alt_stats.ok:
        return alt_mask, alt_stats

    # Both failed: return the border attempt and let QC report it, rather than
    # silently shipping the fallback as though it had succeeded.
    return mask, stats


def _grabcut_refine(bgr: np.ndarray, mask: np.ndarray, iters: int = 3) -> np.ndarray:
    """Refine boundaries with GrabCut, seeded from the threshold mask."""
    gc = np.full(mask.shape, cv2.GC_PR_BGD, np.uint8)
    eroded = cv2.erode(mask, np.ones((9, 9), np.uint8), iterations=2)
    dilated = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=2)
    gc[dilated == 0] = cv2.GC_BGD
    gc[mask > 0] = cv2.GC_PR_FGD
    gc[eroded > 0] = cv2.GC_FGD
    if not (gc == cv2.GC_FGD).any() or not (gc == cv2.GC_BGD).any():
        return mask
    try:
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(bgr, gc, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return mask
    out = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return out if out.mean() > 0 else mask


# ---------------------------------------------------------------------------
# Batch caching
# ---------------------------------------------------------------------------
def mask_path_for(image_path: str | Path, cache: Path = MASK_CACHE) -> Path:
    """Deterministic cache location mirroring <class>/<stem>.png."""
    p = Path(image_path)
    return cache / p.parent.name / f"{p.stem}.png"


def _worker(args: tuple[str, bool, bool]) -> dict:
    """Segment one image. `cached` means the PNG already existed and was kept.

    QC stats are always recomputed rather than assumed. An earlier version
    short-circuited on a cached mask and returned `ok: True` without looking at
    it, so a second run without `--overwrite` produced a `mask_qc.csv` in which
    every image passed -- and `harvest_all` and the P5 ablation both filter on
    exactly that column.
    """
    path, use_grabcut, overwrite = args
    out = mask_path_for(path)
    have = out.exists() and not overwrite

    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return {"path": path, "cached": False, "ok": False, "reason": "imread failed",
                "area_frac": 0.0, "method": "none"}

    mask, stats = segment_leaf(bgr, use_grabcut=use_grabcut)
    if not have:
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), mask)
    return {"path": path, "cached": have, **asdict(stats)}


def build_masks(
    df: pd.DataFrame, use_grabcut: bool = False, overwrite: bool = False, workers: int = 12
) -> pd.DataFrame:
    tasks = [(p, use_grabcut, overwrite) for p in df["path"].tolist()]
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for r in tqdm(pool.map(_worker, tasks), total=len(tasks), unit="img", desc="masks"):
            rows.append(r)
    return pd.DataFrame(rows)


def make_montage(
    df: pd.DataFrame, results: pd.DataFrame, out: Path, n: int = 3, cell: int = 160
) -> None:
    """Grid of image | mask | cutout, `n` samples per class -- the visual QC gate.

    Numbers can hide a systematically inverted or truncated mask; a montage
    cannot.
    """
    merged = df.merge(results[["path", "ok"]], on="path", how="left")
    classes = sorted(merged["class"].unique())
    rows_img = []
    for cls in classes:
        sub = merged[merged["class"] == cls].head(n)
        strip = []
        for _, r in sub.iterrows():
            bgr = cv2.imread(r["path"], cv2.IMREAD_COLOR)
            mp = mask_path_for(r["path"])
            mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if bgr is None or mask is None:
                continue
            bgr = cv2.resize(bgr, (cell, cell))
            mask = cv2.resize(mask, (cell, cell), interpolation=cv2.INTER_NEAREST)
            m3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cut = cv2.bitwise_and(bgr, m3)
            # Magenta border marks a QC failure.
            if not r.get("ok", True):
                for im in (bgr, m3, cut):
                    cv2.rectangle(im, (0, 0), (cell - 1, cell - 1), (255, 0, 255), 3)
            strip += [bgr, m3, cut]
        if strip:
            while len(strip) < n * 3:
                strip.append(np.zeros((cell, cell, 3), np.uint8))
            rows_img.append(np.hstack(strip))

    if not rows_img:
        print("montage: nothing to draw")
        return
    grid = np.vstack(rows_img)
    labelled = np.zeros((grid.shape[0], grid.shape[1] + 260, 3), np.uint8)
    labelled[:, 260:] = grid
    for i, cls in enumerate(classes[: len(rows_img)]):
        cv2.putText(
            labelled, cls[:34], (5, i * cell + cell // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA,
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), labelled)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=Path, default=SPLITS_CSV)
    ap.add_argument("--grabcut", action="store_true", help="slower boundary refinement")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="process only N images (smoke test)")
    args = ap.parse_args()

    ensure_dirs()
    df = load_manifest(args.splits)
    if args.limit:
        df = df.groupby("class", group_keys=False).head(max(1, args.limit // df["class"].nunique()))

    res = build_masks(df, use_grabcut=args.grabcut, overwrite=args.overwrite, workers=args.workers)

    # Stats are recomputed for every row, cached PNG or not, so report over all.
    n_ok = int(res["ok"].sum())
    n_tot = len(res)
    pass_rate = n_ok / n_tot if n_tot else 1.0

    montage = REPORTS_DIR / "mask_qc_montage.jpg"
    make_montage(df, res, montage)

    qc_csv = REPORTS_DIR / "mask_qc.csv"
    res.to_csv(qc_csv, index=False)

    print(f"\nMasks cached under {MASK_CACHE}")
    print(f"QC pass rate: {n_ok}/{n_tot} = {pass_rate:.2%}")
    if n_tot and "method" in res:
        print("Segmentation method used:")
        for method, cnt in res["method"].value_counts().items():
            print(f"  {cnt:6d}  {method}")
    if n_tot and "reason" in res:
        # Group by the leading token: full reason strings embed distinct numbers,
        # so counting them whole yields a list of 1s instead of a distribution.
        bad = res[~res["ok"]]["reason"].str.split().str[0].value_counts().head(8)
        if len(bad):
            print("Top failure reasons:")
            for reason, cnt in bad.items():
                print(f"  {cnt:5d}  {reason.rstrip(':')}")
    if "resid_frac" in res:
        # Leaf tissue escaping the generous mask. Not part of `ok` -- it is what
        # background harvesting and the P5 ablation must filter on themselves.
        r = res["resid_frac"].dropna()
        print("Leaf tissue outside the generous mask (resid_frac):")
        print(f"  p50 {r.quantile(.50):.4f}  p90 {r.quantile(.90):.4f}  "
              f"p99 {r.quantile(.99):.4f}  max {r.max():.4f}")
        for cut in (0.01, 0.02):
            n = int((r > cut).sum())
            print(f"  > {cut:.2f}: {n} images ({n / len(r):.2%}) -- excluded from harvesting")
    print(f"Montage: {montage}")
    print(f"QC csv:  {qc_csv}")


if __name__ == "__main__":
    main()
