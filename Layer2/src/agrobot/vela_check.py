"""Compile a TFLite INT8 model with Arm Vela and check it fits the RA8P1.

This is the phase-0 gate. Vela is the Ethos-U compiler: it partitions the graph
into subgraphs it can run on the NPU and leaves the rest to the Cortex-M85. Two
numbers decide whether the design is viable:

* **NPU offload** -- the share of *operators* Vela accepted (its "NPU operators
  = N (X%)" line). Operators it rejects run on the M85 at a small fraction of
  the NPU's throughput, so a graph that is 60% offloaded is not "mostly fine",
  it is dominated by the 40%.
* **Tensor arena (SRAM) peak** -- must fit the 2 MB the RA8P1 gives us. Vela's
  `sram_memory_used` is the real allocation after its scheduler has done its
  liveness analysis, which is why guessing from layer shapes is not enough.

Not to be confused with offload: `cycles_npu / cycles_total` is the share of
cycles spent computing rather than waiting on memory. Both can look like "a
percentage of NPU utilisation" while meaning different things -- a model can be
100% offloaded and still stall on weight fetches from off-chip flash. Both are
reported, separately labelled.

`--accelerator-config` defaults to `ethos-u55-128` (128 MAC/cycle), the
configuration Renesas documents for the RA8P1. The arena size is passed
explicitly rather than left to Vela's default system config, which assumes 4 MB.

Vela is invoked as a subprocess rather than imported: it is a CLI tool whose
Python entry point is not a stable API, and the CSV summary it writes is the
documented output.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agrobot.paths import EXPORT_DIR, ensure_dirs  # noqa: E402

SRAM_BUDGET_KB = 2048  # RA8P1 tensor arena
FLASH_BUDGET_KB = 8192  # 8 MB part; MRAM variants are 512 KB / 1 MB
MRAM_BUDGET_KB = 1024  # the 1 MB MRAM variant, reported as a bonus
MIN_NPU_OFFLOAD = 0.90  # below this the M85 fallback dominates latency

DEFAULT_ACCELERATOR = "ethos-u55-128"
SRAM_ARENA_BYTES = SRAM_BUDGET_KB * 1024

# TFLite operators the Ethos-U55 can execute, per Vela's SUPPORTED_OPS.md. This
# is the subset relevant to the classifiers here, not the exhaustive list --
# anything absent should be checked against Vela's docs before being added.
# Kept here so a test can assert the exported graph stays inside it without
# having to run the compiler.
VELA_SUPPORTED_OPS = frozenset({
    "ADD", "AVERAGE_POOL_2D", "CONCATENATION", "CONV_2D", "DEPTHWISE_CONV_2D",
    "FULLY_CONNECTED", "HARD_SWISH", "LOGISTIC", "MAX_POOL_2D", "MAXIMUM",
    "MEAN", "MINIMUM", "MUL", "PAD", "QUANTIZE", "RELU", "RELU6", "RESHAPE",
    "RESIZE_BILINEAR", "RESIZE_NEAREST_NEIGHBOR", "SLICE", "SOFTMAX",
    "SPLIT", "SQUEEZE", "STRIDED_SLICE", "SUB", "TANH", "TRANSPOSE",
})


def run_vela(
    model: Path,
    out_dir: Path,
    accelerator: str = DEFAULT_ACCELERATOR,
    optimise: str = "Performance",
    arena_cache_size: int = SRAM_ARENA_BYTES,
) -> tuple[int, str]:
    """Invoke the `vela` CLI. Returns (returncode, combined output).

    `--arena-cache-size` is set explicitly because Vela's default system config
    assumes a 4 MB arena -- twice what the RA8P1 has. Left at the default, its
    scheduler optimises against memory we do not own, and a model that overflows
    2 MB would still be reported as fitting.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    exe = shutil.which("vela")
    cmd = [exe] if exe else [sys.executable, "-m", "ethosu.vela"]
    cmd += [
        str(model),
        "--accelerator-config", accelerator,
        "--optimise", optimise,
        "--arena-cache-size", str(arena_cache_size),
        "--output-dir", str(out_dir),
        "--enable-debug-db",  # writes the per-operator placement database
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def parse_summary_csv(out_dir: Path) -> dict:
    """Read Vela's `*_summary_*.csv`, the authoritative numbers.

    Vela's stdout is human-readable but its field names have shifted between
    releases; the CSV is the stable contract. Keys are kept verbatim so that a
    renamed column shows up as a missing metric rather than a silent zero.
    """
    files = sorted(out_dir.glob("*summary*.csv"))
    if not files:
        return {}
    with files[-1].open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {}
    raw = rows[-1]
    return {k.strip(): v.strip() for k, v in raw.items() if k}


def _num(d: dict, *names: str) -> float | None:
    """Fetch the first present key, tolerating Vela's naming drift."""
    for n in names:
        for key, val in d.items():
            if key.lower() == n.lower():
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return None
    return None


def interpret(summary: dict, stdout: str) -> dict:
    """Turn the CSV into a pass/fail verdict against the RA8P1 budgets."""
    sram_kb = _num(summary, "sram_memory_used")
    flash_kb = _num(summary, "off_chip_flash_memory_used")
    on_chip_kb = _num(summary, "on_chip_flash_memory_used")
    inference_s = _num(summary, "inference_time")
    ips = _num(summary, "inferences_per_second")
    macs = _num(summary, "nn_macs")
    cycles_npu = _num(summary, "cycles_npu")
    cycles_total = _num(summary, "cycles_total")

    # Fall back to stdout when the CSV schema does not match this Vela release.
    if sram_kb is None:
        m = re.search(r"Total SRAM used\s+([\d.]+)\s*KiB", stdout)
        sram_kb = float(m.group(1)) if m else None
    if flash_kb is None:
        m = re.search(r"Total Off-chip Flash used\s+([\d.]+)\s*KiB", stdout)
        flash_kb = float(m.group(1)) if m else None

    # Operator placement is the offload metric that matters: an operator Vela
    # rejects runs on the M85 instead of the NPU. Vela prints it as
    # "CPU operators = 0 (0.0%)" / "NPU operators = 65 (100.0%)".
    def _ops(kind: str) -> int | None:
        m = re.search(rf"{kind} operators?\s*=\s*(\d+)", stdout)
        return int(m.group(1)) if m else None

    n_npu, n_cpu = _ops("NPU"), _ops("CPU")
    offload = None
    if n_npu is not None and n_cpu is not None and (n_npu + n_cpu) > 0:
        offload = n_npu / (n_npu + n_cpu)

    # Distinct from offload, and easy to confuse with it: the share of cycles
    # spent computing rather than waiting on memory. Weights stream from
    # off-chip flash at 0.47 GB/s, so a low value here means the model is
    # bandwidth-starved even at 100% operator offload.
    compute_bound = (
        cycles_npu / cycles_total if cycles_npu is not None and cycles_total else None
    )

    checks = {
        "sram_fits": None if sram_kb is None else sram_kb <= SRAM_BUDGET_KB,
        "flash_fits": None if flash_kb is None else flash_kb <= FLASH_BUDGET_KB,
        "offload_ok": None if offload is None else offload >= MIN_NPU_OFFLOAD,
    }
    # An unmeasured check is a FAILED check. Reporting "gate passed" because a
    # metric could not be parsed is exactly the self-deception this gate exists
    # to prevent.
    return {
        "npu_offload": offload,
        "n_ops_npu": n_npu,
        "n_ops_cpu": n_cpu,
        "sram_kb": sram_kb,
        "flash_kb": flash_kb,
        "on_chip_flash_kb": on_chip_kb,
        "fits_1mb_mram": None if flash_kb is None else flash_kb <= MRAM_BUDGET_KB,
        "inference_ms": None if inference_s is None else inference_s * 1000.0,
        "inferences_per_second": ips,
        "macs": macs,
        "compute_bound_frac": compute_bound,
        "checks": checks,
        "unmeasured": [k for k, v in checks.items() if v is None],
        "passed": all(v is True for v in checks.values()),
    }


def _fmt(v, unit: str = "", nd: int = 1) -> str:
    return "n/a" if v is None else f"{v:,.{nd}f}{unit}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True, help="full-integer .tflite")
    ap.add_argument("--out-dir", type=Path, default=EXPORT_DIR / "vela")
    ap.add_argument("--accelerator", default=DEFAULT_ACCELERATOR)
    ap.add_argument("--optimise", default="Performance", choices=["Performance", "Size"])
    ap.add_argument(
        "--arena-cache-size", type=int, default=SRAM_ARENA_BYTES,
        help=f"SRAM available to the tensor arena in bytes (default {SRAM_ARENA_BYTES})",
    )
    args = ap.parse_args()

    ensure_dirs()
    if not args.model.exists():
        raise SystemExit(f"model not found: {args.model}")

    print(f"Compiling {args.model.name} for {args.accelerator} (optimise={args.optimise}, "
          f"arena={args.arena_cache_size / 1024:.0f} KiB)...")
    rc, output = run_vela(
        args.model, args.out_dir, args.accelerator, args.optimise, args.arena_cache_size
    )
    print(output.strip()[-4000:])
    if rc != 0:
        raise SystemExit(f"\nvela failed with exit code {rc}")

    summary = parse_summary_csv(args.out_dir)
    verdict = interpret(summary, output)
    report = {"model": str(args.model), "accelerator": args.accelerator,
              "optimise": args.optimise, "arena_cache_size": args.arena_cache_size,
              "vela_summary": summary, **verdict}
    (args.out_dir / "vela_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    off = verdict["npu_offload"]
    cb = verdict["compute_bound_frac"]
    print("\n" + "=" * 62)
    print("RA8P1 / Ethos-U55 fit")
    print("=" * 62)
    print(f"  NPU offload      : {'n/a' if off is None else f'{off:.1%}'}"
          f"   (need >= {MIN_NPU_OFFLOAD:.0%})")
    print(f"  ops on NPU / CPU : {verdict['n_ops_npu']} / {verdict['n_ops_cpu']}")
    print(f"  SRAM (arena)     : {_fmt(verdict['sram_kb'], ' KiB')}"
          f"   (budget {SRAM_BUDGET_KB} KiB)")
    print(f"  Flash (weights)  : {_fmt(verdict['flash_kb'], ' KiB')}"
          f"   (budget {FLASH_BUDGET_KB} KiB"
          f"; fits 1 MB MRAM: {verdict['fits_1mb_mram']})")
    print(f"  Inference        : {_fmt(verdict['inference_ms'], ' ms', 2)}"
          f"   ({_fmt(verdict['inferences_per_second'], ' fps', 1)}, "
          f"{_fmt(verdict['macs'], '', 0)} MACs)")
    print(f"  Compute-bound    : {'n/a' if cb is None else f'{cb:.1%}'}"
          "   (NPU cycles / total; the rest is memory wait)")
    for name, ok in verdict["checks"].items():
        print(f"  {name:<16} : {'PASS' if ok else 'UNMEASURED' if ok is None else 'FAIL'}")
    print(f"\nReport: {args.out_dir / 'vela_report.json'}")

    if not verdict["passed"]:
        if verdict["unmeasured"]:
            print(f"\nGATE FAILED -- could not measure: {verdict['unmeasured']}. "
                  "Treating an unmeasured budget as a failure rather than a pass.")
        else:
            print("\nGATE FAILED -- the RA8P1 design needs to change before training.")
        raise SystemExit(2)
    print("\nGATE PASSED -- safe to spend GPU hours on training.")


if __name__ == "__main__":
    main()
