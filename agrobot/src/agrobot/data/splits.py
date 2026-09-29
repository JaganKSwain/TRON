"""Train / val / test splitting that refuses to leak.

Two split products, written as one `split` column in `splits.csv`:

* **train / val / test** -- stratified by class, *grouped* so that no leaf
  identity (and no near-duplicate cluster) appears on two sides. This is the
  headline evaluation.

* **test_site** -- for the 6 classes photographed at more than one capture
  site, an *entire* site is withheld from training and evaluated separately.
  This measures the thing that actually predicts field performance: can the
  model recognise a disease photographed somewhere it has never seen? It costs
  no extra training run because those images are simply routed out of train/val.

A fully site-disjoint split is impossible here -- 8 of 15 classes come from a
single capture session, so withholding their site would delete the class. The
`test_site` subset is the strongest cross-domain signal this dataset can give.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agrobot.data.manifest import load_manifest  # noqa: E402
from agrobot.paths import (  # noqa: E402
    MANIFEST_CSV,
    RANDOM_SEED,
    REPORTS_DIR,
    SPLIT_FRACTIONS,
    SPLITS_CSV,
    ensure_dirs,
)


def choose_holdout_sites(df: pd.DataFrame, min_holdout: int = 100) -> dict[str, str]:
    """Pick one capture site per class to withhold entirely from training.

    Rule: among a class's *non-dominant* sites, take the smallest one with at
    least `min_holdout` images. Smallest keeps the most training data while
    still leaving a large enough sample to measure. Classes with only one
    usable site are skipped.
    """
    holdouts: dict[str, str] = {}
    for cls, sub in df.groupby("class"):
        counts = Counter(sub["site"])
        if len(counts) < 2:
            continue
        ranked = counts.most_common()
        candidates = [(s, n) for s, n in ranked[1:] if n >= min_holdout]
        if not candidates:
            continue
        site = min(candidates, key=lambda t: t[1])[0]
        holdouts[cls] = site
    return holdouts


def _group_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per group: its size and its majority class.

    Groups are usually single-class, but a near-duplicate union can span
    classes; the majority class decides where the group is stratified.
    """
    rows = []
    for gid, sub in df.groupby("group_id"):
        rows.append(
            {
                "group_id": gid,
                "size": len(sub),
                "class": sub["class"].value_counts().index[0],
                "n_classes": sub["class"].nunique(),
            }
        )
    return pd.DataFrame(rows)


def allocate_groups(
    groups: pd.DataFrame,
    fractions: dict[str, float],
    seed: int = RANDOM_SEED,
) -> dict[int, str]:
    """Greedily assign whole groups to splits to hit per-class target fractions.

    Largest-group-first, each group going to whichever split is furthest below
    its target *image* count for that class. Group integrity is absolute; the
    fractions are approximate (they cannot be exact when groups are indivisible).
    """
    assignment: dict[int, str] = {}
    rng_order = groups.sample(frac=1.0, random_state=seed)  # deterministic tie-break

    for cls, sub in rng_order.groupby("class"):
        total = int(sub["size"].sum())
        targets = {k: total * v for k, v in fractions.items()}
        current = dict.fromkeys(fractions, 0.0)
        # Largest first: better packing, keeps small splits from overshooting.
        for _, row in sub.sort_values("size", ascending=False).iterrows():
            deficit = {k: targets[k] - current[k] for k in fractions}
            pick = max(deficit, key=lambda k: (deficit[k], -current[k]))
            assignment[int(row["group_id"])] = pick
            current[pick] += row["size"]
        del cls
    return assignment


def build_splits(
    df: pd.DataFrame,
    fractions: dict[str, float] | None = None,
    min_holdout: int = 100,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, dict]:
    fractions = fractions or SPLIT_FRACTIONS
    df = df.copy()

    # --- Route held-out sites to test_site, group-consistently --------------
    holdouts = choose_holdout_sites(df, min_holdout=min_holdout)
    in_holdout = pd.Series(False, index=df.index)
    for cls, site in holdouts.items():
        in_holdout |= (df["class"] == cls) & (df["site"] == site)

    # If any member of a group is a held-out-site image, the whole group goes
    # with it -- otherwise a near-duplicate could sneak back into train.
    holdout_groups = set(df.loc[in_holdout, "group_id"].unique())
    site_mask = df["group_id"].isin(holdout_groups)
    n_dragged = int(site_mask.sum() - in_holdout.sum())

    df["split"] = ""
    df.loc[site_mask, "split"] = "test_site"

    # --- Split the remainder ------------------------------------------------
    rest = df[~site_mask]
    groups = _group_frame(rest)
    assignment = allocate_groups(groups, fractions, seed=seed)
    df.loc[~site_mask, "split"] = df.loc[~site_mask, "group_id"].map(assignment)

    stats = {
        "holdout_sites": holdouts,
        "n_test_site": int(site_mask.sum()),
        "n_dragged_by_group": n_dragged,
        "counts": df["split"].value_counts().to_dict(),
        "n_multiclass_groups": int((groups["n_classes"] > 1).sum()),
    }
    return df, stats


def verify_splits(df: pd.DataFrame) -> list[str]:
    """Return a list of integrity violations. Empty list means the split is sound."""
    problems: list[str] = []

    # 1. No group may appear in two splits -- the whole point of the exercise.
    per_group = df.groupby("group_id")["split"].nunique()
    straddling = per_group[per_group > 1]
    if len(straddling):
        problems.append(
            f"{len(straddling)} groups straddle splits (e.g. {list(straddling.index[:5])})"
        )

    # 2. No exact-duplicate md5 across splits.
    per_md5 = df.groupby("md5")["split"].nunique()
    if (bad := per_md5[per_md5 > 1]).shape[0]:
        problems.append(f"{len(bad)} md5 hashes appear in more than one split")

    # 3. Every class must be present in train, val and test.
    for split in ("train", "val", "test"):
        present = set(df[df["split"] == split]["class"])
        missing = set(df["class"]) - present
        if missing:
            problems.append(f"split '{split}' is missing classes: {sorted(missing)}")

    # 4. A held-out site must not leak into train or val.
    for cls, sub in df.groupby("class"):
        site_only = set(sub[sub["split"] == "test_site"]["site"])
        leaked = set(sub[sub["split"].isin(["train", "val"])]["site"]) & site_only
        if leaked:
            problems.append(f"class {cls}: held-out site(s) {leaked} also present in train/val")

    return problems


def write_report(df: pd.DataFrame, stats: dict, problems: list[str], out: Path) -> None:
    lines = ["# Split report", ""]

    counts = df["split"].value_counts()
    total = len(df)
    lines += ["## Overall", "", "| Split | Images | Share |", "|---|---:|---:|"]
    for name in ("train", "val", "test", "test_site"):
        n = int(counts.get(name, 0))
        lines.append(f"| {name} | {n} | {n / total:.1%} |")
    lines.append(f"| **total** | **{total}** | |")

    lines += [
        "",
        "## Integrity", "",
        ("**PASS** -- no group, duplicate or held-out site crosses a split boundary."
         if not problems else "**FAIL**"),
    ]
    for p in problems:
        lines.append(f"- {p}")

    lines += [
        "",
        "## Cross-site holdout (`test_site`)", "",
        "These sites were withheld from training entirely. Accuracy here is the best "
        "available proxy for field performance: same disease, unseen capture conditions.",
        "",
        "| Class | Held-out site | Images |",
        "|---|---|---:|",
    ]
    if stats["holdout_sites"]:
        for cls, site in sorted(stats["holdout_sites"].items()):
            n = int(((df["class"] == cls) & (df["site"] == site)).sum())
            lines.append(f"| {cls} | `{site}` | {n} |")
    else:
        lines.append("| _(none found)_ | | |")

    lines += [
        "",
        f"- Images pulled in by group consistency: {stats['n_dragged_by_group']}",
        f"- Groups spanning >1 class (near-duplicate unions): {stats['n_multiclass_groups']}",
        "",
        "The remaining 9 classes have only one capture site, so they cannot contribute "
        "a cross-site measurement -- withholding their site would delete the class.",
        "",
        "## Per-class breakdown", "",
        "| Class | train | val | test | test_site |",
        "|---|---:|---:|---:|---:|",
    ]
    pivot = df.pivot_table(
        index="class", columns="split", values="path", aggfunc="count", fill_value=0
    )
    for cls in sorted(df["class"].unique()):
        row = pivot.loc[cls] if cls in pivot.index else {}
        vals = [int(row.get(s, 0)) for s in ("train", "val", "test", "test_site")]
        lines.append(f"| {cls} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} |")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=MANIFEST_CSV)
    ap.add_argument("--out", type=Path, default=SPLITS_CSV)
    ap.add_argument("--min-holdout", type=int, default=100)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = ap.parse_args()

    ensure_dirs()
    df = load_manifest(args.manifest)
    df, stats = build_splits(df, min_holdout=args.min_holdout, seed=args.seed)
    problems = verify_splits(df)

    df.to_csv(args.out, index=False)
    report = REPORTS_DIR / "split_report.md"
    write_report(df, stats, problems, report)

    print(f"Splits: {args.out}")
    print(f"Report: {report}")
    for k, v in df["split"].value_counts().items():
        print(f"  {k}: {v}")
    print(f"  held-out sites: {stats['holdout_sites']}")

    if problems:
        print("\nINTEGRITY FAILURES:")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)
    print("\nIntegrity checks passed.")


if __name__ == "__main__":
    main()
