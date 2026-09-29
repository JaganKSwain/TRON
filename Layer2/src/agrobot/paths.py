"""Central path and constant configuration.

Code lives under the dataset folder (OneDrive-synced, small). All heavy
artifacts -- venv, checkpoints, mask cache, run logs -- live under WORK_ROOT
which is deliberately OUTSIDE OneDrive: a ~5 GB CUDA venv plus per-epoch
checkpoints would otherwise thrash OneDrive sync.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Data -------------------------------------------------------------------
# The PlantVillage class folders. Override with AGROBOT_DATA_ROOT.
DATA_ROOT = Path(
    os.environ.get(
        "AGROBOT_DATA_ROOT",
        r"C:\Users\jagan\OneDrive\Documents\TRON\Dataset\PlantVillage",
    )
)

# Directory names under DATA_ROOT that are NOT classes.
#   "PlantVillage" -- a byte-identical duplicate of all 15 class folders
#                     (md5-verified). Including it would place every image in
#                     both train and test.
#   "agrobot"      -- this project's own source tree.
NON_CLASS_DIRS = frozenset({"PlantVillage", "agrobot", ".claude", ".git", "__pycache__"})

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

# --- Work tree (outside OneDrive) -------------------------------------------
WORK_ROOT = Path(os.environ.get("AGROBOT_WORK_ROOT", r"C:\ml\agrobot"))

MANIFEST_DIR = WORK_ROOT / "manifest"
MANIFEST_CSV = MANIFEST_DIR / "manifest.csv"
SPLITS_CSV = MANIFEST_DIR / "splits.csv"
MASK_CACHE = WORK_ROOT / "masks"
BACKGROUND_CACHE = WORK_ROOT / "backgrounds"
RUNS_DIR = WORK_ROOT / "runs"
REPORTS_DIR = WORK_ROOT / "reports"
EXPORT_DIR = WORK_ROOT / "export"

# Optional: real AgroBot / field photographs, organised as
# field_val/<class_name>/*.jpg. If present, evaluate.py reports metrics on it
# separately -- this is the ONLY true measurement of deployment accuracy.
FIELD_VAL_DIR = WORK_ROOT / "field_val"

# --- Label space ------------------------------------------------------------
# The 16th label is an open-set reject class synthesised from background crops
# (see data/backgrounds.py). It is NOT a PlantVillage class.
NOT_A_LEAF = "__not_a_leaf__"

# --- Split configuration ----------------------------------------------------
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 1337


def ensure_dirs() -> None:
    """Create the work-tree directories. Safe to call repeatedly."""
    for d in (
        MANIFEST_DIR,
        MASK_CACHE,
        BACKGROUND_CACHE,
        RUNS_DIR,
        REPORTS_DIR,
        EXPORT_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
