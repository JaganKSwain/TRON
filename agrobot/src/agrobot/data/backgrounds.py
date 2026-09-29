"""Background bank for `RandomBackgroundSwap` and the `__not_a_leaf__` class.

This is the highest-leverage augmentation in the project, and the reason is
specific to PlantVillage: 8 of 15 classes come from a single capture session, so
the backdrop *predicts the label*. A CNN can score ~99% by reading the
photo-session and then collapse on real plants (Mohanty 2016: 99.35% lab ->
~31% field). Compositing each leaf onto a randomly chosen backdrop severs that
correlation directly -- it is the difference between a model that recognises
pathology and one that recognises a grey card.

Three sources, in increasing order of field realism:

1. **Harvested backdrops** -- the leaf inpainted out of a real PlantVillage
   image, leaving that session's true backdrop, lighting and shadow. Free, and
   sufficient on its own to decorrelate backdrop from class, since a
   `Tomato_Leaf_Mold` leaf now appears on a `Potato___Early_blight` backdrop.
   But they are all still uniform studio cards, so they do not teach clutter.
2. **Procedural textures** -- soil, gravel, blurred foliage, sky, straw,
   greenhouse concrete. Where actual field diversity comes from: a leaf against
   *other leaves* is the real deployment condition and appears nowhere in
   PlantVillage.
3. **User field photographs** -- anything dropped into `WORK_ROOT/backgrounds/
   field/`. Strictly better than the other two; the harness picks them up
   automatically so adding real AgroBot frames later needs no code change.

Masks are grown by `segment.generous_mask` -- closed, then dilated -- before
harvesting. Masks that pass QC can still have notches: washed-out leaf regions
that read as backdrop and connect to the mask boundary, which `_fill_holes`
cannot close because it reaches them from outside. Without that margin those leaf
pixels end up in the background bank and, worse, in the `__not_a_leaf__` reject
class. Sources are additionally filtered on `resid_frac`, which catches the case
the mask geometry looks fine but the distance map lost a petiole or midrib.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agrobot.data.manifest import load_manifest  # noqa: E402
from agrobot.data.segment import generous_mask, mask_path_for  # noqa: E402
from agrobot.paths import (  # noqa: E402
    BACKGROUND_CACHE,
    REPORTS_DIR,
    SPLITS_CSV,
    ensure_dirs,
)

# Upper bound on leaf tissue left outside the generous mask, as a share of the
# frame, for an image to be harvested from. See `segment.residue_frac`: masks can
# pass QC and still leave a petiole or a specular midrib outside, and those pixels
# would go straight into the background bank and the `__not_a_leaf__` class.
# Measured distribution after the segmentation fix: p50 0.0000, p90 0.0023,
# p99 0.0293, max 0.0931. 0.01 drops the worst ~4% and keeps the rest.
MAX_RESID_FRAC = 0.01

HARVEST_DIR = BACKGROUND_CACHE / "harvested"
FIELD_DIR = BACKGROUND_CACHE / "field"

PROCEDURAL_KINDS = ("soil", "gravel", "foliage", "sky", "straw", "concrete")

# Separator between class name and index in harvested filenames. It must be a
# character the class names cannot contain: they are full of underscores
# (`Pepper__bell___Bacterial_spot`, `Potato___Early_blight`), so splitting on
# "__" yields "Pepper" rather than the class -- which silently disabled
# same-class exclusion, since the lookup key never matched.
CLASS_SEP = "~~"


def harvested_class_of(path: Path | str) -> str:
    """Recover the source class from a harvested backdrop's filename."""
    return Path(path).name.rsplit(CLASS_SEP, 1)[0]


# ---------------------------------------------------------------------------
# 1. Harvesting real backdrops
# ---------------------------------------------------------------------------
def harvest_backdrop(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """Remove the leaf and inpaint the hole, yielding a full-frame backdrop.

    Inpainting is used rather than cropping a leaf-free corner because it keeps
    the frame's real lighting gradient, vignetting and cast shadow. PlantVillage
    backdrops are smooth cards, so propagating from the boundary produces a
    plausible empty backdrop rather than the smear it would give on a busy scene.

    The mask is grown by `segment.generous_mask` -- closed, then dilated -- so
    notches and thin petioles cannot leave leaf pixels behind. The same function
    backs `segment.residue_frac` and the P5 ablation, so all three agree on what
    "the leaf, generously" means.

    Returns None when the grown leaf leaves too little backdrop to propagate
    from -- a frame-filling leaf has no backdrop worth harvesting.
    """
    grown = generous_mask(mask)
    if float((grown > 0).mean()) > 0.80:
        return None
    return cv2.inpaint(bgr, (grown > 0).astype(np.uint8), 7, cv2.INPAINT_TELEA)


def _harvest_worker(args: tuple[str, str]) -> str | None:
    path, out_name = args
    out = HARVEST_DIR / out_name
    if out.exists():
        return str(out)
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path_for(path)), cv2.IMREAD_GRAYSCALE)
    if bgr is None or mask is None:
        return None
    bg = harvest_backdrop(bgr, mask)
    if bg is None:
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), bg, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return str(out)


def harvest_all(
    df: pd.DataFrame,
    qc: pd.DataFrame | None = None,
    per_class: int = 40,
    workers: int = 12,
) -> list[str]:
    """Harvest up to `per_class` backdrops from each class.

    Balanced per class on purpose: the bank must not be dominated by the largest
    class, or the most common backdrop would still carry label information and
    the swap would only partly break the shortcut.

    Sources are filtered twice, because `ok` and `resid_frac` answer different
    questions. `ok` excludes masks whose geometry is untrustworthy -- inverted,
    partial, fragmented. `resid_frac` excludes masks that are geometrically fine
    but still leave leaf tissue outside the generous mask; `ok` cannot see that,
    since `leak_frac` is derived from the same distance map that lost those pixels.
    """
    src = df
    if qc is not None and "ok" in qc.columns:
        good = set(qc.loc[qc["ok"].astype(bool), "path"])
        src = src[src["path"].isin(good)]
        print(f"  {len(src):,} of {len(df):,} images have a QC-passing mask")

    if qc is not None and "resid_frac" in qc.columns:
        clean = set(qc.loc[qc["resid_frac"].fillna(1.0) <= MAX_RESID_FRAC, "path"])
        before = len(src)
        src = src[src["path"].isin(clean)]
        print(f"  {len(src):,} of those leave no leaf tissue outside the mask "
              f"(resid_frac <= {MAX_RESID_FRAC}; dropped {before - len(src):,})")
    else:
        print("  WARNING: no resid_frac column -- cannot exclude masks that leak "
              "leaf tissue into the bank; re-run agrobot.data.segment")

    # Only harvest from training images: a backdrop lifted from a val/test photo
    # would put that photo's session into the training distribution.
    if "split" in src.columns:
        src = src[src["split"] == "train"]
        print(f"  {len(src):,} of those are in the train split")

    tasks: list[tuple[str, str]] = []
    for cls, grp in src.groupby("class"):
        for i, p in enumerate(grp["path"].head(per_class)):
            tasks.append((p, f"{cls}{CLASS_SEP}{i:03d}.jpg"))

    HARVEST_DIR.mkdir(parents=True, exist_ok=True)
    out: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for r in tqdm(pool.map(_harvest_worker, tasks), total=len(tasks),
                      unit="bg", desc="harvest"):
            if r:
                out.append(r)
    return out


# ---------------------------------------------------------------------------
# 2. Procedural textures
# ---------------------------------------------------------------------------
def _fractal_noise(
    h: int,
    w: int,
    rng: np.random.Generator,
    octaves: int = 5,
    persistence: float = 0.55,
    base: int = 2,
) -> np.ndarray:
    """Value noise summed over octaves, contrast-stretched to [0, 1].

    Cheap stand-in for Perlin: upsample small random grids bicubically and add
    them with decaying amplitude. Multi-scale structure is what makes a texture
    read as soil or foliage rather than as television static.

    `base` is the coarsest grid; the finest is `base * 2**(octaves-1)`, capped at
    the image size. It matters more than it looks. With the old hard-wired
    `base=2`, six octaves topped out at a 64x64 grid stretched over 256px -- no
    structure within 4px of pixel scale -- and that octave carried only
    `persistence**5` = 3% of the amplitude. Measured Laplacian variance of the
    output was 2.6 and Canny found no edges at all: `soil`, `sky` and the
    `concrete` base were smooth colour ramps, contributing nothing over a solid
    fill. Reaching pixel scale needs `base * 2**(octaves-1) >= max(h, w)`.

    The result is stretched on 2nd/98th percentiles. Summed octaves pile up near
    the mean -- measured std 0.09 of the range -- so without the stretch `_ramp`
    interpolates only across the middle of its two colours and every texture came
    out as its own flat mid-tone, whatever endpoints the caller asked for.
    """
    acc = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        # Capped: a grid finer than the image would be cubic-*downsampled*, which
        # aliases. At the cap an octave is pixel-scale grain, which is wanted.
        res = int(min(max(h, w), max(2, base * 2 ** o)))
        grid = rng.random((res, res), dtype=np.float32)
        acc += cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC) * amp
        total += amp
        amp *= persistence
    out = acc / max(total, 1e-6)
    lo, hi = (float(v) for v in np.percentile(out, (2.0, 98.0)))
    if hi - lo < 1e-6:  # degenerate field; nothing to stretch
        return np.clip(out, 0.0, 1.0)
    return np.clip((out - lo) / (hi - lo), 0.0, 1.0)


def _ramp(noise: np.ndarray, lo: tuple[int, int, int], hi: tuple[int, int, int]) -> np.ndarray:
    """Colourise a scalar field by interpolating between two BGR colours."""
    n = noise[..., None]
    a = np.array(lo, np.float32)
    b = np.array(hi, np.float32)
    return np.clip(a + (b - a) * n, 0, 255).astype(np.uint8)


def procedural_background(
    size: int = 256, rng: np.random.Generator | None = None, kind: str | None = None
) -> np.ndarray:
    """One synthetic backdrop in BGR. `kind` defaults to a random choice."""
    rng = rng or np.random.default_rng()
    kind = kind or str(rng.choice(PROCEDURAL_KINDS))
    h = w = size
    s = size / 256.0  # densities and feature sizes below are tuned at 256px

    if kind == "soil":
        n = _fractal_noise(h, w, rng, octaves=6, persistence=0.6)
        img = _ramp(n, (28, 40, 58), (96, 122, 150))
    elif kind == "gravel":
        # Drawn, not noised. `_fractal_noise` starts from a 2x2 grid and doubles,
        # so at 2-3 octaves it is a handful of frame-sized blobs: thresholding it
        # produced smooth continents with no pebbles in them. What makes gravel
        # read as gravel is many *discrete* stones separated by dark contact
        # shadows -- high-frequency structure the octave scheme cannot reach at
        # any affordable persistence. So draw the stones.
        img = _ramp(_fractal_noise(h, w, rng, octaves=5, persistence=0.5),
                    (34, 38, 44), (74, 80, 88))  # packed dirt between stones
        for _ in range(int(rng.integers(260, 460) * s * s)):
            cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
            ra = max(2, int(round(float(rng.integers(3, 12)) * s)))
            rb = max(2, int(round(ra * rng.uniform(0.55, 1.0))))
            ang = float(rng.integers(0, 180))
            # One grey per stone, jittered only slightly per channel. Drawing the
            # three channels independently -- the mistake in `straw` below -- turns
            # gravel into coloured confetti.
            v = float(rng.integers(88, 205))
            col = tuple(float(c) for c in np.clip(v + rng.uniform(-6, 6, 3), 0, 255))
            cv2.ellipse(img, (cx, cy), (ra + 1, rb + 1), ang, 0, 360,
                        (18, 20, 24), -1)  # crevice shadow, so stones separate
            cv2.ellipse(img, (cx, cy), (ra, rb), ang, 0, 360, col, -1)
            if rng.random() < 0.45:  # specular cap, lit from upper left
                cv2.ellipse(img, (cx - ra // 3, cy - rb // 3),
                            (max(1, ra // 3), max(1, rb // 3)), ang, 0, 360,
                            tuple(min(255.0, c + 42) for c in col), -1)
        grit = rng.normal(0, 7, (h, w, 1)).astype(np.float32)
        img = np.clip(img.astype(np.float32) + grit, 0, 255).astype(np.uint8)
        img = cv2.GaussianBlur(img, (3, 3), 0.7)
    elif kind == "foliage":
        # The important one: in the field a leaf is surrounded by other leaves.
        img = np.zeros((h, w, 3), np.uint8)
        img[:] = (38, 70, 34)
        for _ in range(int(rng.integers(40, 90))):
            centre = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            axes = (int(rng.integers(12, 60)), int(rng.integers(8, 40)))
            colour = (int(rng.integers(20, 70)), int(rng.integers(55, 145)),
                      int(rng.integers(15, 65)))
            cv2.ellipse(img, centre, axes, float(rng.integers(0, 180)),
                        0, 360, colour, -1)
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=float(rng.uniform(1.5, 5.0)))
    elif kind == "sky":
        # Leaves photographed from below, against bright sky -- a lighting
        # regime with no analogue in the dataset.
        grad = np.linspace(1.0, 0.35, h, dtype=np.float32)[:, None].repeat(w, 1)
        img = _ramp(grad, (150, 130, 105), (250, 246, 238))
        cloud = _fractal_noise(h, w, rng, octaves=4)
        img = np.clip(img + (cloud[..., None] - 0.5) * 40, 0, 255).astype(np.uint8)
    elif kind == "straw":
        # Straw is one hue at many brightnesses. The previous version sampled B, G
        # and R independently, which decorrelates them: a draw of (108, 96, 203)
        # is magenta and (55, 160, 135) is green, so the mulch came out as
        # confetti. Pick a tan and scale it instead -- the channels must move
        # together for the hue to survive.
        TAN = np.array([88, 138, 178], np.float32)  # BGR, mid-tone straw
        img = _ramp(_fractal_noise(h, w, rng, octaves=5, persistence=0.5),
                    (52, 78, 100), (96, 132, 164))  # crushed litter beneath
        # Clumped, not uniform. Fallen straw lies in local drifts that share an
        # orientation; independent uniform angles average out into felt.
        for _ in range(int(rng.integers(5, 9))):
            ox, oy = rng.uniform(0, w), rng.uniform(0, h)
            base = float(rng.uniform(0, np.pi))
            for _ in range(int(rng.integers(10, 26))):
                cx = ox + rng.normal(0, w * 0.13)
                cy = oy + rng.normal(0, h * 0.13)
                ang = base + float(rng.normal(0, 0.28))
                ln = float(rng.uniform(0.10, 0.34)) * size
                dx, dy = np.cos(ang) * ln / 2, np.sin(ang) * ln / 2
                # Three points, not two: a slight bend reads as a stalk rather
                # than as a ruled line.
                pts = np.array([
                    (cx - dx, cy - dy),
                    (cx + rng.normal(0, 2), cy + rng.normal(0, 2)),
                    (cx + dx, cy + dy),
                ], np.int32)
                th = max(1, int(round(rng.uniform(1.0, 3.2) * s)))
                k = float(rng.uniform(0.55, 1.25))  # bleached <-> damp
                col = tuple(float(c) for c in
                            np.clip(TAN * k + rng.uniform(-8, 8, 3), 0, 255))
                cv2.polylines(img, [pts], False, (26, 34, 42), th + 1, cv2.LINE_AA)
                cv2.polylines(img, [pts], False, col, th, cv2.LINE_AA)
        img = cv2.GaussianBlur(img, (3, 3), 0.6)
    else:  # concrete -- greenhouse and shed floors
        n = _fractal_noise(h, w, rng, octaves=5, persistence=0.4)
        img = _ramp(n, (118, 120, 124), (186, 190, 194))
        speck = rng.random((h, w), dtype=np.float32)
        img = np.clip(img + ((speck > 0.985)[..., None] * 45), 0, 255).astype(np.uint8)

    # Every real photograph has a lighting gradient; a perfectly even backdrop is
    # itself a cue the model could latch onto.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = rng.uniform(0, w), rng.uniform(0, h)
    fall = 1.0 - 0.30 * np.sqrt(((xx - cx) / w) ** 2 + ((yy - cy) / h) ** 2)
    img = np.clip(img.astype(np.float32) * fall[..., None], 0, 255).astype(np.uint8)
    return img


# ---------------------------------------------------------------------------
# 3. The bank
# ---------------------------------------------------------------------------
class BackgroundBank:
    """Samples backdrops for `RandomBackgroundSwap`.

    Files are read from disk on demand rather than held in memory: a DataLoader
    with 8 workers would otherwise carry 8 copies of the bank, and decoding a
    256x256 JPEG costs well under a millisecond.

    `p_procedural` controls the mix. It defaults high because harvested
    backdrops, while real, are all uniform studio cards -- they decorrelate
    backdrop from label but teach nothing about clutter. Procedural textures are
    the only source here that resembles a field.
    """

    def __init__(
        self,
        harvest_dir: Path = HARVEST_DIR,
        field_dir: Path = FIELD_DIR,
        p_procedural: float = 0.5,
        p_field: float = 0.5,
    ) -> None:
        self.harvested = sorted(harvest_dir.glob("*.jpg")) if harvest_dir.exists() else []
        self.field = (
            [p for p in sorted(field_dir.rglob("*")) if p.suffix.lower() in
             {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
            if field_dir.exists() else []
        )
        self.p_procedural = p_procedural
        self.p_field = p_field
        # class name -> harvested files, so a leaf is never composited onto a
        # backdrop lifted from its own class.
        self._by_class: dict[str, list[Path]] = {}
        for p in self.harvested:
            self._by_class.setdefault(harvested_class_of(p), []).append(p)
        # Pre-built "everything except this class" pools. Filtering per sample
        # would cost a scan of the whole bank on every training image.
        self._excluding: dict[str, list[Path]] = {
            cls: [p for p in self.harvested if harvested_class_of(p) != cls]
            for cls in self._by_class
        }

    def __len__(self) -> int:
        return len(self.harvested) + len(self.field)

    def describe(self) -> str:
        return (
            f"BackgroundBank(harvested={len(self.harvested)} across "
            f"{len(self._by_class)} classes, field={len(self.field)}, "
            f"procedural={len(PROCEDURAL_KINDS)} kinds, "
            f"p_procedural={self.p_procedural}, p_field={self.p_field})"
        )

    def sample(
        self,
        size: int = 256,
        rng: np.random.Generator | None = None,
        exclude_class: str | None = None,
    ) -> np.ndarray:
        """One backdrop, BGR uint8, `size` x `size`.

        Never returns None: if no files exist the procedural generator always
        can, so training does not silently lose the augmentation because a
        harvest step was skipped.
        """
        rng = rng or np.random.default_rng()

        # Real field photographs beat everything else, so give them their own
        # first-class share of the draw when any are present.
        if self.field and rng.random() < self.p_field:
            img = self._read(self.field[int(rng.integers(len(self.field)))])
            if img is not None:
                return self._fit(img, size, rng)

        if self.harvested and rng.random() >= self.p_procedural:
            pool = self._excluding.get(exclude_class) or self.harvested
            img = self._read(pool[int(rng.integers(len(pool)))])
            if img is not None:
                return self._fit(img, size, rng)

        return procedural_background(size, rng)

    @staticmethod
    def _read(path: Path) -> np.ndarray | None:
        return cv2.imread(str(path), cv2.IMREAD_COLOR)

    @staticmethod
    def _fit(img: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
        """Random crop to a square then resize, so a backdrop is never reused
        pixel-for-pixel across epochs."""
        h, w = img.shape[:2]
        side = min(h, w)
        if side > size:
            side = int(rng.integers(size, side + 1))
        y = int(rng.integers(0, max(1, h - side + 1)))
        x = int(rng.integers(0, max(1, w - side + 1)))
        crop = img[y:y + side, x:x + side]
        return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def make_contact_sheet(bank: BackgroundBank, out: Path, cols: int = 8, rows: int = 6,
                       cell: int = 128, seed: int = 0) -> None:
    """Grid of sampled backdrops -- the visual check that none contain a leaf."""
    rng = np.random.default_rng(seed)
    tiles = []
    for _ in range(cols * rows):
        tiles.append(cv2.resize(bank.sample(256, rng), (cell, cell)))
    grid = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid)


def make_kind_sheet(out: Path, cell: int = 160, per_kind: int = 5, seed: int = 3) -> None:
    """One row per procedural kind, labelled."""
    rng = np.random.default_rng(seed)
    rows = []
    for kind in PROCEDURAL_KINDS:
        strip = [cv2.resize(procedural_background(256, rng, kind), (cell, cell))
                 for _ in range(per_kind)]
        row = np.hstack(strip)
        pad = np.zeros((cell, 150, 3), np.uint8)
        cv2.putText(pad, kind, (6, cell // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(np.hstack([pad, row]))
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=Path, default=SPLITS_CSV)
    ap.add_argument("--per-class", type=int, default=40)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    ensure_dirs()
    df = load_manifest(args.splits)
    qc_csv = REPORTS_DIR / "mask_qc.csv"
    qc = pd.read_csv(qc_csv) if qc_csv.exists() else None
    if qc is None:
        print("WARNING: no mask_qc.csv -- harvesting from unvetted masks")

    print(f"Harvesting up to {args.per_class} backdrops per class...")
    got = harvest_all(df, qc, per_class=args.per_class, workers=args.workers)
    print(f"  harvested {len(got)} backdrops -> {HARVEST_DIR}")

    bank = BackgroundBank()
    print(f"\n{bank.describe()}")
    if not bank.field:
        print(f"  (drop real field photos into {FIELD_DIR} to improve this)")

    sheet = REPORTS_DIR / "background_bank.jpg"
    kinds = REPORTS_DIR / "background_kinds.jpg"
    make_contact_sheet(bank, sheet)
    make_kind_sheet(kinds)
    print(f"\nContact sheet : {sheet}")
    print(f"Procedural kinds: {kinds}")


if __name__ == "__main__":
    main()
