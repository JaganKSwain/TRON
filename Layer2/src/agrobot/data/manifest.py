"""Build a verified, de-duplicated manifest of the PlantVillage dataset.

Three problems in this dataset make a naive `ImageFolder` scan dangerous, and
this module exists to handle all three explicitly:

1. `<DATA_ROOT>/PlantVillage/` is a byte-identical duplicate of all 15 class
   folders. Scanning it doubles every image, so a random split would place the
   same file in both train and test.

2. Many images are repeat shots of the *same physical leaf*: `GHLB Leaf 2`
   appears 16 times (Days 1-16), `GH_HL Leaf 495` 4 times. Filenames encode
   this, so we parse a `leaf_core` key and later split on it -- otherwise Day 6
   lands in train and Day 9 in test.

3. Capture-site codes correlate almost perfectly with the label (8 of 15
   classes come from a single photo session). We record `site` so the shortcut
   can be measured rather than silently exploited.

Outputs `manifest.csv` with one row per surviving image, plus a Markdown report.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Allow `python -m agrobot.data.manifest` and direct script execution.
if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agrobot.paths import (  # noqa: E402
    DATA_ROOT,
    IMAGE_EXTENSIONS,
    MANIFEST_CSV,
    NON_CLASS_DIRS,
    REPORTS_DIR,
    ensure_dirs,
)

# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------
# PlantVillage names look like:
#   <uuid>___<site> <leaf>[.<shot>][ Day <n>].<ext>
# Examples:
#   00fc2ee5-...___RS_HL 1864.JPG          -> site RS_HL,  leaf "RS_HL 1864"
#   000146ff-...___GH_HL Leaf 259.1.JPG    -> site GH_HL,  leaf "GH_HL Leaf 259", shot 1
#   d0badc95-...___GHLB Leaf 10 Day 6.jpeg -> site GHLB,   leaf "GHLB Leaf 10",   day 6
#   ...___GHLB Leaf 2.1 Day 16.JPG         -> site GHLB,   leaf "GHLB Leaf 2", shot 1, day 16
_DAY_RE = re.compile(r"\s+Day\s+(\d+)\s*$", re.IGNORECASE)
_SHOT_RE = re.compile(r"\.(\d+)$")

# Pearson r between normalised 32x32 thumbnails, above which a hash-proposed
# pair is accepted as a genuine near-duplicate.
#
# Measured on this dataset (see `--calibrate`), correlation of:
#   * the same photo recompressed to q75, shifted 2px, brightened 5%
#     -> min 0.870, p05 0.917, median 0.955      <- must be KEPT
#   * random cross-class pairs (n=3000)
#     -> max 0.837, p99 0.653                    <- must be REJECTED
# 0.90 sits in that gap. It is deliberately closer to the negative ceiling than
# the positive floor: a missed near-duplicate costs a little test optimism,
# whereas a false cross-class union merges two labels into one split group and
# corrupts stratification.
#
# Note the same leaf shot from a *different angle* correlates only 0.10-0.40 and
# is not recoverable this way -- dhash never proposes those as candidates, and
# `leaf_core` filename grouping is what actually catches them.
MIN_DUP_CORR = 0.90

# Thumbnails sit beside the manifest; only this module and its calibration need them.
THUMBS_NPY = MANIFEST_CSV.parent / "thumbs.npy"


@dataclass(frozen=True)
class ParsedName:
    uuid: str
    site: str
    leaf_core: str  # descriptor with " Day N" and a trailing ".N" removed
    shot: str
    day: str
    descriptor: str


def parse_filename(name: str) -> ParsedName:
    """Split a PlantVillage filename into capture-site and leaf identity.

    `leaf_core` is the grouping key: repeat shots of one leaf (``.1``, ``.2``)
    and a leaf tracked across days (``Day 6``, ``Day 9``) collapse to the same
    value, so they can be kept on one side of a train/test split.
    """
    stem = Path(name).stem

    if "___" in stem:
        uuid_part, descriptor = stem.split("___", 1)
    else:
        # A handful of web-sourced files (e.g. "svn-r6Yb5c") carry no uuid.
        uuid_part, descriptor = "", stem
    descriptor = descriptor.strip()

    core = descriptor
    day = ""
    if (m := _DAY_RE.search(core)) is not None:
        day = m.group(1)
        core = core[: m.start()]

    shot = ""
    if (m := _SHOT_RE.search(core)) is not None:
        shot = m.group(1)
        core = core[: m.start()]
    core = core.strip()

    tokens = core.split()
    site = tokens[0] if tokens else (descriptor or "unknown")

    return ParsedName(
        uuid=uuid_part,
        site=site,
        leaf_core=core or site,
        shot=shot,
        day=day,
        descriptor=descriptor,
    )


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------
def file_md5(path: Path, chunk: int = 1 << 20) -> str:
    """Exact-content hash, used to drop true duplicates."""
    h = hashlib.md5()  # noqa: S324 - dedupe only, not security
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def dhash(img: Image.Image, size: int = 8) -> int:
    """64-bit difference hash, used only to *generate candidates*.

    Implemented directly (rather than via `imagehash`) to avoid a scipy
    dependency. Resize to (size+1, size) greyscale and compare horizontally
    adjacent pixels; the sign pattern is robust to mild changes in exposure,
    compression and scale -- exactly the differences between repeat shots of
    one leaf.

    Hamming distance alone is not decisive here. Every image is a centred leaf
    on a flat backdrop, so at distance <= 4 the hash proposes pairs spanning
    different species and capture sites (measured: 3 of 36 candidates). Every
    candidate is therefore confirmed against pixels by `confirm_pairs` before it
    is allowed to affect the split.
    """
    small = img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    arr = np.asarray(small, dtype=np.int16)
    bits = (arr[:, 1:] > arr[:, :-1]).flatten()
    out = 0
    for bit in bits:
        out = (out << 1) | int(bit)
    return out


# dhash is stored as fixed-width hex, never as an integer: values above 2^53
# are silently rounded when a CSV round-trips them through float64.
def dhash_to_hex(value: int) -> str:
    return f"{value:016x}"


def hashes_from_hex(series) -> np.ndarray:
    return np.array([int(s, 16) for s in series], dtype=np.uint64)


def load_manifest(path: Path = MANIFEST_CSV) -> pd.DataFrame:
    """Read the manifest with dtypes that survive the round-trip.

    `dhash` must be read as str: as an integer it exceeds 2^53 and pandas
    parses the column to float64, silently rounding away the low bits. `shot`
    and `day` must be str too, or "1" and "" mix into a float column.
    """
    return pd.read_csv(path, dtype={"dhash": str, "shot": str, "day": str}).fillna(
        {"shot": "", "day": ""}
    )


THUMB_SIZE = 32


def thumbnail(img: Image.Image, size: int = THUMB_SIZE) -> np.ndarray:
    """Small greyscale thumbnail retained for pixel-level duplicate confirmation."""
    small = img.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(small, dtype=np.uint8)


def _popcount64(x: np.ndarray) -> np.ndarray:
    """Population count over a uint64 array."""
    if hasattr(np, "bitwise_count"):  # numpy >= 2.0
        return np.bitwise_count(x).astype(np.int16)
    # Fallback: byte-wise table lookup.
    table = np.array([bin(i).count("1") for i in range(256)], dtype=np.int16)
    view = x.view(np.uint8).reshape(-1, 8)
    return table[view].sum(axis=1).astype(np.int16)


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------
def discover_class_dirs(root: Path) -> list[Path]:
    """Class folders under `root`, excluding the nested duplicate and non-data dirs.

    A directory counts as a class only if it holds at least one file with a real
    image extension. Two traps this guards against:

    * **Tool directories.** `.pytest_cache`, `.ipynb_checkpoints` and friends
      appear if anything is ever run from the dataset root, and silently become a
      16th class -- shifting every label index. Hidden directories are skipped
      outright rather than enumerated in `NON_CLASS_DIRS`, which can only ever
      list the ones already encountered.
    * **Extensionless files.** An earlier version also accepted `p.suffix == ""`
      as an image, on the theory that some PlantVillage files lack extensions.
      The dataset's only such file is `svn-r6Yb5c`, which is **zero bytes** and
      does not decode -- while `.gitignore` inside a cache directory matched the
      clause and made that directory look like a class.
    """
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if d.name in NON_CLASS_DIRS or d.name.startswith("."):
            continue
        has_image = any(
            p.suffix.lower() in IMAGE_EXTENSIONS for p in d.iterdir() if p.is_file()
        )
        if has_image:
            out.append(d)
    return out


def _probe(args: tuple[Path, str]) -> dict | None:
    """Read one image: hash it, verify it decodes, record its shape."""
    path, class_name = args
    try:
        md5 = file_md5(path)
        with Image.open(path) as img:
            img.load()  # force decode -- catches truncated JPEGs
            width, height = img.size
            mode = img.mode
            dh = dhash(img)
            thumb = thumbnail(img)
    except Exception as exc:  # noqa: BLE001 - want every failure reason recorded
        return {"path": str(path), "class": class_name, "error": f"{type(exc).__name__}: {exc}"}

    parsed = parse_filename(path.name)
    return {
        "path": str(path),
        "filename": path.name,
        "class": class_name,
        "site": parsed.site,
        "leaf_core": parsed.leaf_core,
        "shot": parsed.shot,
        "day": parsed.day,
        "uuid": parsed.uuid,
        "md5": md5,
        "dhash": dhash_to_hex(dh),
        "width": width,
        "height": height,
        "mode": mode,
        "error": "",
        # Carried on the row so it stays aligned through filtering and dedupe;
        # split out into a separate array before the CSV is written.
        "_thumb": thumb.tobytes(),
    }


def _thumbs_array(df: pd.DataFrame) -> np.ndarray:
    """Stack the per-row thumbnails into (N, THUMB_SIZE, THUMB_SIZE) uint8."""
    if not len(df):
        return np.zeros((0, THUMB_SIZE, THUMB_SIZE), np.uint8)
    return np.stack(
        [np.frombuffer(b, dtype=np.uint8).reshape(THUMB_SIZE, THUMB_SIZE) for b in df["_thumb"]]
    )


def scan(root: Path = DATA_ROOT, workers: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scan all class folders. Returns (ok_rows, failed_rows)."""
    class_dirs = discover_class_dirs(root)
    if not class_dirs:
        raise RuntimeError(f"No class directories with images found under {root}")

    tasks: list[tuple[Path, str]] = []
    for d in class_dirs:
        for p in sorted(d.iterdir()):
            if not p.is_file():
                continue
            if p.suffix.lower() in IMAGE_EXTENSIONS or p.suffix == "":
                tasks.append((p, d.name))

    print(f"Scanning {len(tasks)} files across {len(class_dirs)} classes...")
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for row in tqdm(pool.map(_probe, tasks), total=len(tasks), unit="img"):
            if row is not None:
                rows.append(row)

    df = pd.DataFrame(rows)
    failed = df[df["error"] != ""].drop(columns=["_thumb"], errors="ignore").copy()
    ok = df[df["error"] == ""].copy().reset_index(drop=True)
    return ok, failed


# ---------------------------------------------------------------------------
# Grouping (union-find over leaf identity + near-duplicate clusters)
# ---------------------------------------------------------------------------
class _DisjointSet:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _near_duplicate_pairs(
    hashes: np.ndarray, max_distance: int, bucket_cap: int = 4000
) -> tuple[list[tuple[int, int]], dict]:
    """Candidate pairs whose dhash differs by <= max_distance bits.

    Uses multi-index hashing: split each 64-bit hash into 8 bytes and only
    compare images that agree exactly on at least one byte. If two hashes
    differ in `d` bits those bits touch at most `d` bytes, so at least `8 - d`
    bytes match exactly -- the search is exhaustive for any `d <= 7` while
    avoiding the 2*10^8 comparisons a brute-force scan over 20k images needs.

    `bucket_cap` guards against a low-entropy byte position producing one
    enormous bucket and a quadratic blow-up. On this dataset it never triggers
    (every byte position has 255+ distinct values, largest bucket 5.2%), but if
    it ever does, the returned `exhaustive` flag says whether the pigeonhole
    guarantee still holds: skipping `s` whole chunks is safe only while
    `8 - max_distance > s`.
    """
    n = len(hashes)
    if n == 0:
        return [], {"skipped_buckets": 0, "skipped_chunks": 0, "exhaustive": True}
    h64 = hashes.astype(np.uint64)
    byte_view = h64.view(np.uint8).reshape(n, 8)

    pairs: set[tuple[int, int]] = set()
    skipped_buckets = 0
    skipped_chunks: set[int] = set()
    for chunk in range(8):
        buckets: dict[int, list[int]] = defaultdict(list)
        for idx, val in enumerate(byte_view[:, chunk]):
            buckets[int(val)].append(idx)
        for members in buckets.values():
            m = len(members)
            if m < 2:
                continue
            if m > bucket_cap:
                skipped_buckets += 1
                skipped_chunks.add(chunk)
                continue
            idxs = np.asarray(members)
            block = h64[idxs]
            # Pairwise XOR then popcount, upper triangle only.
            xor = block[:, None] ^ block[None, :]
            dist = _popcount64(xor.ravel()).reshape(m, m)
            iu = np.triu_indices(m, k=1)
            hit = dist[iu] <= max_distance
            for a, b in zip(idxs[iu[0]][hit], idxs[iu[1]][hit]):
                pairs.add((int(min(a, b)), int(max(a, b))))

    # Pigeonhole: >= (8 - max_distance) bytes match, so at least one match
    # survives outside the skipped chunks iff (8 - max_distance) > |skipped|.
    guaranteed = (8 - max_distance) > len(skipped_chunks)
    stats = {
        "skipped_buckets": skipped_buckets,
        "skipped_chunks": len(skipped_chunks),
        "exhaustive": bool(guaranteed),
    }
    return sorted(pairs), stats


def _normalise_thumbs(thumbs: np.ndarray) -> np.ndarray:
    """Z-score each thumbnail so comparison ignores exposure/brightness offsets."""
    flat = thumbs.reshape(len(thumbs), -1).astype(np.float32)
    flat -= flat.mean(axis=1, keepdims=True)
    std = flat.std(axis=1, keepdims=True)
    std[std < 1e-6] = 1.0  # a perfectly flat image has no structure to compare
    return flat / std


def confirm_pairs(
    pairs: list[tuple[int, int]], thumbs: np.ndarray, min_corr: float
) -> tuple[list[tuple[int, int]], list[float]]:
    """Keep only candidate pairs whose thumbnails actually correlate.

    A 64-bit dhash collides on this dataset: leaf-shaped blobs centred on a
    flat backdrop look alike to it even across species and capture sites.
    Correlating normalised 32x32 thumbnails settles it on pixels. Candidates
    are few (hundreds), so this costs nothing.
    """
    if not pairs:
        return [], []
    z = _normalise_thumbs(thumbs)
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    corr = (z[a] * z[b]).mean(axis=1)  # Pearson r, thumbnails already z-scored
    keep = corr >= min_corr
    return [pairs[i] for i in np.flatnonzero(keep)], corr.tolist()


def calibrate_dup_threshold(
    df: pd.DataFrame, thumbs: np.ndarray, near_dup_distance: int = 4, seed: int = 0
) -> dict:
    """Measure where the duplicate-confirmation threshold belongs.

    Compares three populations of pairs so `MIN_DUP_CORR` is chosen from data:

    * **same-shot** -- same leaf, same day, different shot index. Frames taken
      seconds apart: the strongest available ground truth for "duplicate".
    * **random cross-class** -- different species/disease. Ground-truth negatives.
    * **hash candidates** -- what `_near_duplicate_pairs` proposes, which is the
      population the threshold actually has to separate.
    """
    z = _normalise_thumbs(thumbs)

    def corr(pairs: list[tuple[int, int]]) -> np.ndarray:
        if not pairs:
            return np.array([])
        a = np.array([p[0] for p in pairs])
        b = np.array([p[1] for p in pairs])
        return (z[a] * z[b]).mean(axis=1)

    # Positives: same leaf_core + same day, differing only in shot index.
    by_key: dict[tuple, list[int]] = defaultdict(list)
    days = df["day"].fillna("").astype(str)
    shots = df["shot"].fillna("").astype(str)
    for i, (cls, leaf, day, shot) in enumerate(
        zip(df["class"], df["leaf_core"], days, shots)
    ):
        if shot:
            by_key[(cls, leaf, day)].append(i)
    pos = [
        (m[i], m[j])
        for m in by_key.values()
        if len(m) > 1
        for i in range(len(m))
        for j in range(i + 1, len(m))
    ]

    # Negatives: random pairs from different classes.
    rng = np.random.default_rng(seed)
    cls_arr = df["class"].to_numpy()
    neg: list[tuple[int, int]] = []
    while len(neg) < 3000:
        a, b = int(rng.integers(len(df))), int(rng.integers(len(df)))
        if a != b and cls_arr[a] != cls_arr[b]:
            neg.append((a, b))

    cand, _ = _near_duplicate_pairs(hashes_from_hex(df["dhash"]), near_dup_distance)

    c_pos, c_neg, c_cand = corr(pos), corr(neg), corr(cand)
    out: dict = {
        "n_same_shot_pairs": len(pos),
        "same_shot_corr_min": float(c_pos.min()) if len(c_pos) else float("nan"),
        "same_shot_corr_p05": float(np.percentile(c_pos, 5)) if len(c_pos) else float("nan"),
        "same_shot_corr_median": float(np.median(c_pos)) if len(c_pos) else float("nan"),
        "cross_class_corr_max": float(c_neg.max()) if len(c_neg) else float("nan"),
        "cross_class_corr_p99": float(np.percentile(c_neg, 99)) if len(c_neg) else float("nan"),
        "n_candidates": len(cand),
        "candidate_corr_median": float(np.median(c_cand)) if len(c_cand) else float("nan"),
    }
    if len(c_cand):
        for t in (0.80, 0.85, 0.90, 0.92, 0.95, 0.98):
            kept = [p for p, c in zip(cand, c_cand) if c >= t]
            xc = sum(1 for a, b in kept if cls_arr[a] != cls_arr[b])
            out[f"kept_at_{t:.2f}"] = f"{len(kept)} pairs, {xc} cross-class"
    return out



def assign_groups(
    df: pd.DataFrame,
    thumbs: np.ndarray,
    near_dup_distance: int = 4,
    min_corr: float = MIN_DUP_CORR,
) -> tuple[pd.DataFrame, dict]:
    """Attach a `group_id` that no train/test split may straddle.

    Two images share a group if they are repeat shots of one leaf (same
    class + `leaf_core`) OR they are *confirmed* near-duplicates. Merging both
    relations with union-find means visually identical frames cannot end up on
    opposite sides of the split, whatever the filenames say.
    """
    df = df.reset_index(drop=True)
    ds = _DisjointSet(len(df))

    # Relation 1: same class + same leaf identity.
    by_leaf: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, (cls, leaf) in enumerate(zip(df["class"], df["leaf_core"])):
        by_leaf[(cls, leaf)].append(i)
    for members in by_leaf.values():
        for j in members[1:]:
            ds.union(members[0], j)
    n_leaf_groups = len(by_leaf)

    # Relation 2: near-duplicate images (any class), hash-proposed then
    # pixel-confirmed. Unconfirmed candidates are discarded: unioning a Pepper
    # leaf with a Tomato leaf would corrupt class stratification.
    candidates, mih_stats = _near_duplicate_pairs(
        hashes_from_hex(df["dhash"]), near_dup_distance
    )
    confirmed, corrs = confirm_pairs(candidates, thumbs, min_corr)

    cross_class_candidates = sum(
        1 for a, b in candidates if df.at[a, "class"] != df.at[b, "class"]
    )
    cross_class_confirmed = sum(
        1 for a, b in confirmed if df.at[a, "class"] != df.at[b, "class"]
    )
    for a, b in confirmed:
        ds.union(a, b)

    roots = [ds.find(i) for i in range(len(df))]
    remap = {r: k for k, r in enumerate(sorted(set(roots)))}
    df["group_id"] = [remap[r] for r in roots]

    stats = {
        "n_leaf_groups": n_leaf_groups,
        "n_dup_candidates": len(candidates),
        "n_dup_candidates_cross_class": cross_class_candidates,
        "n_dup_confirmed": len(confirmed),
        "n_dup_confirmed_cross_class": cross_class_confirmed,
        "dup_min_corr": min_corr,
        "n_final_groups": df["group_id"].nunique(),
        **mih_stats,
    }
    return df, stats


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_manifest(
    root: Path = DATA_ROOT,
    near_dup_distance: int = 4,
    min_corr: float = MIN_DUP_CORR,
    drop_conflicts: bool = True,
) -> tuple[pd.DataFrame, dict]:
    ok, failed = scan(root)

    # Drop exact duplicates (keep the first path per md5).
    before = len(ok)
    dup_mask = ok.duplicated(subset=["md5"], keep="first")
    exact_dups = ok[dup_mask].copy()
    ok = ok[~dup_mask].reset_index(drop=True)

    # An identical file appearing under two different labels is a data problem
    # worth surfacing, not just a duplicate.
    md5_to_classes = exact_dups.merge(
        ok[["md5", "class"]].rename(columns={"class": "kept_class"}), on="md5", how="left"
    )
    conflicting = md5_to_classes[md5_to_classes["class"] != md5_to_classes["kept_class"]]

    # Thumbnails are extracted after dedupe so they stay row-aligned.
    thumbs = _thumbs_array(ok)
    ok = ok.drop(columns=["_thumb"])

    ok, group_stats = assign_groups(
        ok, thumbs, near_dup_distance=near_dup_distance, min_corr=min_corr
    )

    # Same leaf filed under two diseases: one label is wrong. Flag, and by
    # default drop -- training on a known-wrong label injects avoidable noise.
    conflicts = flag_label_conflicts(ok)
    ok["label_conflict"] = False
    if len(conflicts):
        ok.loc[conflicts["index"].tolist(), "label_conflict"] = True
    if len(conflicts) and drop_conflicts:
        keep = ~ok["label_conflict"]
        ok = ok[keep].reset_index(drop=True)
        thumbs = thumbs[np.flatnonzero(keep.to_numpy())]

    stats = {
        "n_scanned": before + len(failed),
        "n_failed": len(failed),
        "n_exact_duplicates_dropped": int(dup_mask.sum()),
        "n_cross_label_duplicates": len(conflicting),
        "n_label_conflicts": len(conflicts),
        "label_conflicts_dropped": bool(drop_conflicts and len(conflicts)),
        "n_images": len(ok),
        "n_classes": ok["class"].nunique(),
        **group_stats,
    }
    return ok, {
        "stats": stats,
        "failed": failed,
        "conflicting": conflicting,
        "conflicts": conflicts,
        "exact_dups": exact_dups.drop(columns=["_thumb"], errors="ignore"),
        "thumbs": thumbs,
    }


def flag_label_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    """Find images whose label disagrees with the rest of their group.

    After union-find, a group is one physical leaf (plus confirmed
    near-duplicates). A group spanning two classes therefore means the *same
    leaf* is filed under two diseases -- one of the labels must be wrong.

    Deciding which: score each candidate class by how many images that class
    draws from this image's capture site, dataset-wide. Capture site is a strong
    provenance signal here (`GH_HL` = greenhouse healthy: 584 healthy images vs
    2 late-blight), so the class with far weaker site affinity is the misfile.
    Returns the suspect rows with the evidence attached.
    """
    rows = []
    site_class = df.groupby(["site", "class"]).size()

    for gid, sub in df.groupby("group_id"):
        if sub["class"].nunique() < 2:
            continue
        # Affinity of each class in this group to the sites the group was shot at.
        affinity: dict[str, int] = {}
        for cls in sub["class"].unique():
            affinity[cls] = int(
                sum(int(site_class.get((site, cls), 0)) for site in sub["site"].unique())
            )
        winner = max(affinity, key=lambda c: affinity[c])
        for idx, r in sub[sub["class"] != winner].iterrows():
            rows.append(
                {
                    "index": idx,
                    "path": r["path"],
                    "filename": r["filename"],
                    "labelled": r["class"],
                    "group_majority": winner,
                    "site": r["site"],
                    "leaf_core": r["leaf_core"],
                    "affinity_labelled": affinity[r["class"]],
                    "affinity_winner": affinity[winner],
                    "group_id": gid,
                }
            )
    return pd.DataFrame(rows)


def write_report(df: pd.DataFrame, meta: dict, out_path: Path) -> None:
    stats = meta["stats"]
    lines: list[str] = ["# PlantVillage dataset report", ""]

    lines += [
        "## Scan summary", "",
        f"- Files scanned: **{stats['n_scanned']}**",
        f"- Unreadable/corrupt: **{stats['n_failed']}**",
        f"- Exact duplicates dropped (md5): **{stats['n_exact_duplicates_dropped']}**",
        f"- Duplicates found under a *different* label: **{stats['n_cross_label_duplicates']}**",
        f"- Same-leaf label conflicts: **{stats['n_label_conflicts']}**"
        f"{' (dropped)' if stats['label_conflicts_dropped'] else ' (kept)'}",
        f"- Images in manifest: **{stats['n_images']}** across **{stats['n_classes']}** classes",
        "",
        "## Grouping (leakage control)", "",
        f"- Leaf identities (class + leaf_core): **{stats['n_leaf_groups']}**",
        f"- Near-duplicate candidates (dhash <= {stats.get('dup_distance', 4)} bits): "
        f"**{stats['n_dup_candidates']}** "
        f"({stats['n_dup_candidates_cross_class']} cross-class)",
        f"- Confirmed by pixel correlation (r >= {stats['dup_min_corr']}): "
        f"**{stats['n_dup_confirmed']}** "
        f"({stats['n_dup_confirmed_cross_class']} cross-class)",
        f"- **Final split groups: {stats['n_final_groups']}**",
        "",
        f"- Candidate search exhaustive: **{stats['exhaustive']}** "
        f"({stats['skipped_chunks']} of 8 hash chunks skipped as oversized)",
    ]
    lines += [
        "",
        "A train/val/test split must never straddle a group: repeat shots of one leaf "
        "(`Leaf 2 Day 6` vs `Leaf 2 Day 9`) and near-identical frames are forced onto "
        "the same side.",
        "",
        "Hash-proposed pairs are confirmed against pixels before they are allowed to "
        "merge groups. A 64-bit dhash is not decisive on this dataset -- every image is "
        "a centred leaf on a flat backdrop, so unrelated species collide -- and an "
        "unconfirmed cross-class union would corrupt class stratification.",
        "",
        "## Per-class composition", "",
        "| Class | Images | Groups | Capture sites (count) |",
        "|---|---:|---:|---|",
    ]

    for cls in sorted(df["class"].unique()):
        sub = df[df["class"] == cls]
        sites = Counter(sub["site"])
        top = ", ".join(f"`{s}` {n}" for s, n in sites.most_common(4))
        if len(sites) > 4:
            top += f", +{len(sites) - 4} more"
        lines.append(f"| {cls} | {len(sub)} | {sub['group_id'].nunique()} | {top} |")

    # The shortcut, quantified: how predictable is the label from the site alone?
    lines += [
        "",
        "## Capture-site shortcut", "",
        "If one capture site dominates a class, background and lighting alone predict "
        "the label -- and a CNN will learn that instead of pathology.",
        "",
        "| Class | Dominant site | Share |",
        "|---|---|---:|",
    ]
    single_site = 0
    for cls in sorted(df["class"].unique()):
        sub = df[df["class"] == cls]
        site, n = Counter(sub["site"]).most_common(1)[0]
        share = n / len(sub)
        if share >= 0.999:
            single_site += 1
        lines.append(f"| {cls} | `{site}` | {share:.1%} |")

    n_classes = df["class"].nunique()
    # Upper bound on accuracy obtainable from the site label alone.
    site_only = (
        df.groupby("site")["class"].agg(lambda s: s.value_counts().iloc[0]).sum() / len(df)
    )
    lines += [
        "",
        f"- Classes sourced from a **single** site: **{single_site} / {n_classes}**",
        f"- Accuracy achievable from the capture site alone: **{site_only:.1%}** "
        f"(chance is {1 / n_classes:.1%})",
        "",
        "That figure is the size of the shortcut available to the model. Breaking it is "
        "what `data/segment.py` + `RandomBackgroundSwap` and the background-only "
        "ablation in `diagnose.py` are for.",
        "",
        "## Image properties", "",
        f"- Resolutions: {dict(Counter(zip(df['width'], df['height'])).most_common(5))}",
        f"- Colour modes: {dict(Counter(df['mode']))}",
    ]

    if len(meta["failed"]):
        lines += ["", "## Unreadable files", ""]
        for _, r in meta["failed"].iterrows():
            lines.append(f"- `{r['path']}` -- {r['error']}")

    conflicts = meta.get("conflicts")
    if conflicts is not None and len(conflicts):
        lines += [
            "", "## Label conflicts (same leaf, two diseases)", "",
            "Detected structurally: these images share a group -- i.e. the same physical "
            "leaf, confirmed by pixel correlation -- with images carrying a different "
            "label. The `group_majority` column is the class whose dataset-wide affinity "
            "to this capture site is far stronger, so the row's own label is the misfile.",
            "",
            "| File | Labelled | Should be | Site | Site affinity (labelled vs correct) |",
            "|---|---|---|---|---:|",
        ]
        for _, r in conflicts.iterrows():
            lines.append(
                f"| `{r['filename'][-34:]}` | {r['labelled']} | {r['group_majority']} "
                f"| `{r['site']}` | {r['affinity_labelled']} vs {r['affinity_winner']} |"
            )
        verb = "excluded from" if stats["label_conflicts_dropped"] else "retained in"
        lines += ["", f"These rows are **{verb}** the manifest (`--keep-conflicts` to change)."]

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DATA_ROOT)
    ap.add_argument("--near-dup-distance", type=int, default=4)
    ap.add_argument("--min-corr", type=float, default=MIN_DUP_CORR)
    ap.add_argument("--out", type=Path, default=MANIFEST_CSV)
    ap.add_argument(
        "--keep-conflicts",
        action="store_true",
        help="keep images whose label conflicts with the rest of their leaf group",
    )
    ap.add_argument(
        "--calibrate",
        action="store_true",
        help="report the correlation separation used to pick --min-corr, then exit",
    )
    args = ap.parse_args()

    ensure_dirs()
    df, meta = build_manifest(
        args.root,
        near_dup_distance=args.near_dup_distance,
        min_corr=args.min_corr,
        drop_conflicts=not args.keep_conflicts,
    )

    if args.calibrate:
        cal = calibrate_dup_threshold(df, meta["thumbs"], args.near_dup_distance)
        print("\nDuplicate-threshold calibration:")
        for k, v in cal.items():
            print(f"  {k}: {v if isinstance(v, str) else f'{v:.4f}'}")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    np.save(THUMBS_NPY, meta["thumbs"])
    report = REPORTS_DIR / "dataset_report.md"
    write_report(df, meta, report)

    print(f"\nManifest: {args.out}  ({len(df)} images)")
    print(f"Thumbs:   {THUMBS_NPY}  {meta['thumbs'].shape}")
    print(f"Report:   {report}")
    for k, v in meta["stats"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
