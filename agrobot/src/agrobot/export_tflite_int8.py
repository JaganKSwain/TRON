"""PyTorch -> ONNX -> TFLite INT8, the RA8P1 half of the export chain.

Run this *before* spending GPU hours. If the chain cannot produce a fully
integer-quantised graph that Vela accepts, the MCU architecture has to change,
and it is much cheaper to learn that from random weights than after training.

The awkward step is ONNX's NCHW versus TFLite's NHWC. `onnx2tf` rewrites the
graph rather than wrapping it in transposes, which is what keeps the result
NPU-friendly -- a stray `TRANSPOSE` between every convolution would run on the
M85 and dominate the latency.

Quantisation is *full* integer: float inputs/outputs would insert QUANTIZE and
DEQUANTIZE at the boundary and, more importantly, signal that intermediate
tensors might be float too. `summarise_tflite` therefore reports every operator
and every tensor dtype, so "it converted" is never mistaken for "it will run on
the NPU".
"""
from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agrobot.models.mcu import MCU_INPUT_SIZE, build_mcu_model  # noqa: E402
from agrobot.paths import EXPORT_DIR, ensure_dirs  # noqa: E402

ONNX_OPSET = 17
INPUT_NAME = "images"
OUTPUT_NAME = "logits"


def export_onnx(
    model: torch.nn.Module, out: Path, size: int = MCU_INPUT_SIZE, simplify: bool = True
) -> Path:
    """Trace to ONNX at a fixed batch size of 1 (the MCU never batches)."""
    model = model.eval()
    dummy = torch.randn(1, 3, size, size)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(out),
        opset_version=ONNX_OPSET,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamo=False,  # the legacy tracer produces the flatter graph onnx2tf prefers
    )

    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnxsim_simplify

            model_simp, ok = onnxsim_simplify(onnx.load(str(out)))
            if ok:
                onnx.save(model_simp, str(out))
            else:
                print("  onnxsim: validation failed, keeping unsimplified graph")
        except Exception as exc:  # noqa: BLE001 - simplification is optional
            print(f"  onnxsim skipped: {type(exc).__name__}: {exc}")
    return out



def get_dataset_samples(n: int = 100, size: int = MCU_INPUT_SIZE) -> np.ndarray:
    try:
        from agrobot.data.manifest import load_manifest
        from agrobot.data.transforms import get_eval_transforms
        from agrobot.paths import SPLITS_CSV
        import cv2

        df = load_manifest(SPLITS_CSV)
        train_df = df[df["split"] == "train"].sample(n=min(n, len(df)), random_state=42).reset_index(drop=True)
        t = get_eval_transforms(size)
        samples = []
        for _, r in train_df.iterrows():
            bgr = cv2.imread(r["path"])
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            res = t(image=rgb)["image"]
            samples.append(res.permute(1, 2, 0).numpy())
        if samples:
            return np.stack(samples, axis=0).astype(np.float32)
    except Exception as e:
        print(f"  Note: falling back to synthetic calibration ({e})")
    rng = np.random.default_rng(0)
    return rng.normal(0.0, 1.0, (n, size, size, 3)).astype(np.float32)

def write_calibration_npy(
    out: Path, data: np.ndarray | None = None, n: int = 100, size: int = MCU_INPUT_SIZE
) -> Path:
    """Calibration tensor in **NHWC**, the layout of the *converted* model.

    This is a trap worth spelling out: the ONNX graph is NCHW, but onnx2tf feeds
    this array straight into the TFLite representative dataset without
    transposing it (`onnx2tf.py`, the `custom_input_op_name_np_data_path`
    branch). Handing it NCHW does not raise -- it produces quantisation ranges
    computed over the wrong axes, and the resulting accuracy loss looks exactly
    like "INT8 is lossy" rather than like a bug.

    Values must already be normalised the way the model sees them at inference,
    because the export passes mean=0/std=1 and the generator only computes
    `(data - mean) / std`.

    For the de-risk pass random data is fine: operator coverage and arena size
    do not depend on the calibration *values*. Accuracy does, so the real export
    passes a sample drawn from the augmented training distribution.
    """
    if data is None:
        rng = np.random.default_rng(0)
        data = rng.random((n, size, size, 3), dtype=np.float32)
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 4 or data.shape[-1] != 3:
        raise ValueError(f"calibration data must be NHWC with 3 channels, got {data.shape}")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, data)
    return out


# onnx2tf validates its conversion by running ONNX and TF side by side on a
# sample batch, which it downloads from GitHub on first use. That download hands
# `np.load` a pickled array, which numpy >= 2 refuses, so the call dies before
# reaching the conversion. Seeding a plain float32 file of the exact expected
# name skips the download and keeps the validation -- which is worth keeping,
# since per-operator ONNX/TF comparison is how NCHW/NHWC transpose bugs surface.
_SAMPLE_DATA_NAME = "calibration_image_sample_data_20x128x128x3_float32.npy"


def _seed_onnx2tf_sample_data(work_dir: Path) -> None:
    target = work_dir / _SAMPLE_DATA_NAME
    if target.exists():
        return
    rng = np.random.default_rng(1)
    np.save(target, rng.random((20, 128, 128, 3), dtype=np.float32))


def onnx_to_tflite_int8(
    onnx_path: Path, out_dir: Path, calib_npy: Path, size: int = MCU_INPUT_SIZE
) -> Path:
    """Convert to a fully int8 TFLite model. Returns the full-integer artifact."""
    import onnx2tf

    out_dir.mkdir(parents=True, exist_ok=True)
    _seed_onnx2tf_sample_data(out_dir)

    # onnx2tf resolves its sample-data cache relative to the process cwd, so run
    # from out_dir to keep every incidental artifact inside the export folder.
    with contextlib.chdir(out_dir):
        onnx2tf.convert(
            input_onnx_file_path=str(onnx_path.resolve()),
            output_folder_path=str(out_dir.resolve()),
            output_integer_quantized_tflite=True,
            quant_type="per-channel",  # Vela allows per-axis only for conv/FC
            # mean 0 / std 1: the calibration tensor is already normalised, so the
            # representative-dataset generator passes it through unchanged.
            custom_input_op_name_np_data_path=[
                [INPUT_NAME, str(calib_npy.resolve()), 0.0, 1.0]
            ],
            copy_onnx_input_output_names_to_tflite=True,
            non_verbose=True,
        )

    candidates = sorted(out_dir.glob("*_full_integer_quant.tflite"))
    if not candidates:
        raise RuntimeError(
            f"onnx2tf produced no full-integer model in {out_dir}; "
            f"found: {[p.name for p in out_dir.glob('*.tflite')]}"
        )
    return candidates[0]


def summarise_tflite(path: Path) -> dict:
    """Report operators and dtypes -- the evidence that this is really INT8."""
    import tensorflow as tf

    # Without this the interpreter attaches XNNPACK and reports a synthetic
    # `DELEGATE` node, which would show up as an operator the graph does not
    # actually contain. We want the graph as Vela will see it.
    interp = tf.lite.Interpreter(
        model_path=str(path),
        experimental_op_resolver_type=(
            tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        ),
    )
    interp.allocate_tensors()

    ops = Counter()
    for op in interp._get_ops_details():  # noqa: SLF001 - no public API for this
        ops[op["op_name"]] += 1

    dtypes = Counter()
    for d in interp.get_tensor_details():
        dtypes[np.dtype(d["dtype"]).name] += 1

    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    float_tensors = sum(v for k, v in dtypes.items() if k.startswith("float"))
    return {
        "file": path.name,
        "size_kb": round(path.stat().st_size / 1024, 1),
        "ops": dict(ops.most_common()),
        "n_ops": sum(ops.values()),
        "tensor_dtypes": dict(dtypes),
        "n_float_tensors": float_tensors,
        "input": {"name": inp["name"], "shape": inp["shape"].tolist(),
                  "dtype": np.dtype(inp["dtype"]).name},
        "output": {"name": out["name"], "shape": out["shape"].tolist(),
                   "dtype": np.dtype(out["dtype"]).name},
    }


def verify_fidelity(
    onnx_path: Path,
    tflite_path: Path,
    n: int = 32,
    size: int = MCU_INPUT_SIZE,
    seed: int = 7,
) -> dict:
    """Compare FP32 ONNX against INT8 TFLite on identical inputs.

    Two different failures show up here, and they look nothing alike:

    * A **layout** bug (NCHW fed where NHWC was expected, or a transpose lost in
      conversion) destroys the correlation entirely -- logits become unrelated,
      r near 0. This is the failure the de-risk pass is hunting, and random
      weights detect it perfectly well.
    * **Quantisation** loss leaves the correlation high but shifts values
      slightly. With random weights the magnitude of that shift is not
      meaningful, so it is reported without a pass/fail threshold; the number
      that matters is measured on real data once the model is trained (P7).
    """
    import onnxruntime as ort
    import tensorflow as tf

    nhwc = get_dataset_samples(n=n, size=size)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_in = sess.get_inputs()[0].name

    interp = tf.lite.Interpreter(
        model_path=str(tflite_path),
        experimental_op_resolver_type=(
            tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        ),
    )
    interp.allocate_tensors()
    inp, outp = interp.get_input_details()[0], interp.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = outp["quantization"]

    fp32_rows, int8_rows = [], []
    for i in range(n):
        one = nhwc[i : i + 1]
        # Both graphs are fixed at batch 1 -- the MCU never batches -- so feed
        # them one sample at a time rather than reshaping the input.
        fp32_rows.append(
            sess.run(None, {onnx_in: one.transpose(0, 3, 1, 2).copy()})[0][0]
        )
        q = np.clip(np.round(one / in_scale + in_zp), -128, 127).astype(inp["dtype"])
        interp.set_tensor(inp["index"], q)
        interp.invoke()
        raw = interp.get_tensor(outp["index"]).astype(np.float32)[0]
        int8_rows.append((raw - out_zp) * out_scale)

    fp32 = np.stack(fp32_rows)
    int8_logits = np.stack(int8_rows)

    per_sample_r = [
        float(np.corrcoef(fp32[i], int8_logits[i])[0, 1]) for i in range(n)
    ]
    agree = float(np.mean(fp32.argmax(1) == int8_logits.argmax(1)))
    return {
        "n": n,
        "logit_pearson_r_mean": float(np.mean(per_sample_r)),
        "logit_pearson_r_min": float(np.min(per_sample_r)),
        "top1_agreement": agree,
        "max_abs_logit_diff": float(np.abs(fp32 - int8_logits).max()),
        "input_quant": {"scale": float(in_scale), "zero_point": int(in_zp)},
        "output_quant": {"scale": float(out_scale), "zero_point": int(out_zp)},
        "layout_ok": bool(np.mean(per_sample_r) > 0.90),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-classes", type=int, default=16)
    ap.add_argument("--size", type=int, default=MCU_INPUT_SIZE)
    ap.add_argument("--ckpt", type=Path, default=None, help="omit for the de-risk pass")
    ap.add_argument("--out-dir", type=Path, default=EXPORT_DIR / "mcu")
    args = ap.parse_args()

    ensure_dirs()
    out_dir: Path = args.out_dir
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building {args.num_classes}-class MCU model at {args.size}x{args.size}...")
    model = build_mcu_model(args.num_classes, pretrained=args.ckpt is None)
    if args.ckpt:
        state = torch.load(args.ckpt, map_location="cpu")
        state_dict = state.get("student_state", state.get("model_state", state.get("model", state)))
        model.load_state_dict(state_dict)
        print(f"  loaded weights from {args.ckpt}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params:,}  (~{n_params / 1e6:.2f} MB as INT8)")

    onnx_path = out_dir / "mcu.onnx"
    print("Exporting ONNX...")
    export_onnx(model, onnx_path, size=args.size)
    print(f"  {onnx_path.name}  {onnx_path.stat().st_size / 1024:.0f} KB")

    real_data = get_dataset_samples(n=100, size=args.size)
    calib = write_calibration_npy(out_dir / "calibration.npy", data=real_data, size=args.size)
    print(f"Converting to TFLite INT8 (calibration {calib.name})...")
    tflite = onnx_to_tflite_int8(onnx_path, out_dir, calib, size=args.size)

    summary = summarise_tflite(tflite)
    print("Verifying FP32 ONNX vs INT8 TFLite on identical inputs...")
    fidelity = verify_fidelity(onnx_path, tflite, size=args.size)
    summary["fidelity"] = fidelity
    (out_dir / "tflite_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nTFLite: {tflite}")
    print(f"  size: {summary['size_kb']} KB, {summary['n_ops']} ops")
    print(f"  input : {summary['input']}")
    print(f"  output: {summary['output']}")
    print(f"  operators: {summary['ops']}")
    print(f"  tensor dtypes: {summary['tensor_dtypes']}")
    if summary["n_float_tensors"]:
        print(f"  NOTE: {summary['n_float_tensors']} float tensors remain")
    print(f"  logit Pearson r: mean {fidelity['logit_pearson_r_mean']:.4f}, "
          f"min {fidelity['logit_pearson_r_min']:.4f}")
    print(f"  top-1 agreement: {fidelity['top1_agreement']:.1%}   "
          f"(random weights -- layout check, not an accuracy claim)")
    print(f"  layout_ok: {fidelity['layout_ok']}")
    print(f"\nNext: vela_check.py --model {tflite}")


if __name__ == "__main__":
    main()
