# P0 — Export-chain de-risk (RA8P1 / Ethos-U55)

**Verdict: PASS.** The MobileNetV2-0.5 architecture converts to a fully
integer TFLite graph that Arm Vela places **entirely** on the Ethos-U55, well
inside the RA8P1's memory budgets. Training can proceed without risk of
discovering an unshippable model afterwards.

Run with random weights on purpose — this phase tests the *toolchain and
architecture*, not accuracy. Reproduce with:

```bash
python src/agrobot/export_tflite_int8.py && python src/agrobot/vela_check.py --model C:/ml/agrobot/export/mcu/mcu_full_integer_quant.tflite
```

## Chain

`PyTorch → ONNX (opset 17) → onnx2tf → TFLite INT8 → vela`

| Stage | Result |
|---|---|
| Parameters | 708,176 |
| ONNX (FP32) | 2,768 KB |
| TFLite (INT8) | 961 KB |
| Vela `.vela` command stream | `mcu_full_integer_quant_vela.tflite` |

## Quantisation is genuinely integer

| Property | Value |
|---|---|
| Input | `int8`, `[1, 160, 160, 3]` (NHWC) |
| Output | `int8`, `[1, 16]` (logits) |
| Tensor dtypes | 124 × `int8`, 58 × `int32`, **0 float** |

The `int32` tensors are biases and shape constants, which is where Ethos-U
expects them. A single float tensor would have meant a dequantise/requantise
boundary mid-graph and a CPU fallback.

## Operator coverage — the reason the architecture was chosen

| Operator | Count | Ethos-U55 |
|---|---:|---|
| `CONV_2D` | 35 | ✅ |
| `DEPTHWISE_CONV_2D` | 17 | ✅ (channel multiplier 1) |
| `ADD` | 10 | ✅ (residual connections) |
| `PAD` | 5 | ✅ |
| `MEAN` | 1 | ✅ (global average pool) |
| `FULLY_CONNECTED` | 1 | ✅ (classifier) |

Two absences matter as much as the presences:

* **No `TRANSPOSE`.** `onnx2tf` rewrote NCHW→NHWC structurally instead of
  wrapping every convolution in a layout shuffle. Transposes would have run on
  the M85 and dominated the runtime.
* **No `CLIP`.** ReLU6 was fused into each convolution's activation function
  rather than emitted as a separate op.

## Vela placement and fit

```
CPU operators = 0 (0.0%)
NPU operators = 65 (100.0%)
```

| Budget | Used | Available | Headroom |
|---|---:|---:|---:|
| SRAM (tensor arena) | **402.1 KiB** | 2048 KiB | 80% free |
| Flash (weights) | **786.3 KiB** | 8192 KiB | 90% free |

Flash usage also fits the **1 MB MRAM** variant (786 < 1024 KiB), so the design
is not tied to the external-flash parts.

The 80% SRAM headroom settles an open question from the plan: **160×160 input
stands, and the 128×128 fallback is not needed.** Peak arena came in at a fifth
of budget, not the ~600 KB estimated.

## Throughput

| Metric | Value |
|---|---:|
| Estimated inference | 3.82 ms |
| Throughput | 261.6 inferences/s |
| MACs | 48,956,480 |
| Compute-bound fraction | 78.7% |

"Compute-bound fraction" is `cycles_npu / cycles_total` — the share of cycles
spent computing rather than waiting on memory. It is **not** the same as
operator offload, though both render as a percentage: this model is 100%
offloaded *and* spends 21% of its cycles waiting on weight fetches from
off-chip flash (0.47 GB/s). Moving weights to MRAM would recover most of that.

This throughput is what makes P9's plan viable: tiled multi-scale inference over
~20 tiles lands near 13 fps, so whole-plant scanning is affordable rather than
aspirational.

### Caveat on the timing number

Vela had no board-specific config and fell back to `Ethos_U55_High_End_Embedded`
/ `Shared_Sram` (500 MHz core, generic embedded bandwidths). So **3.82 ms is
indicative, not a measurement of this board.** The offload and memory-fit
figures do not depend on that profile and are solid; the latency should be
re-measured on hardware via RUHMI. Arena size was overridden to the RA8P1's real
2 MB rather than Vela's 4 MB default, so the scheduler optimised against memory
we actually have.

## Conversion fidelity

FP32 ONNX vs INT8 TFLite on 32 identical inputs:

| Metric | Value |
|---|---:|
| Logit Pearson r (mean) | 0.9894 |
| Logit Pearson r (min) | 0.9785 |
| Top-1 agreement | 90.6% |

The correlation is the meaningful figure here, and it confirms no layout
scrambling — a NCHW/NHWC mix-up would have driven r to ≈0. The 90.6% top-1
agreement is **not** an accuracy result: with random weights all 16 logits sit
nearly tied, so argmax flips on quantisation noise alone. The real FP32→INT8
delta gets measured on trained weights and real images in P7.

## Bugs this phase caught

Worth recording, because both would have silently degraded the shipped model:

1. **Calibration layout.** `onnx2tf` feeds the calibration array straight into
   the TFLite representative dataset *without transposing it*. Supplying NCHW
   (the ONNX layout, the intuitive choice) raises no error — it computes
   quantisation ranges over the wrong axes. The damage would have looked exactly
   like "INT8 is lossy on small models" and been accepted as normal.
2. **A gate that passed when it measured nothing.** The first `vela_check.py`
   parsed offload with regexes matching a different Vela version, got `None`,
   and printed *"GATE PASSED"* on the strength of the two checks that did parse.
   It would have waved through a 40%-offloaded model. Unmeasured now counts as
   failure, and offload is read from operator placement rather than from
   `cycles_npu/cycles_total`, which measures something else entirely.

## Environment

| Package | Version |
|---|---|
| torch | 2.13.0+cu126 |
| onnx | 1.22.0 |
| onnx2tf | 1.28.8 |
| tensorflow | 2.21.0 |
| ethos-u-vela | 5.1.0 |
| onnxruntime | 1.29.0 |
| numpy | 2.4.6 |

`onnx2tf` under-declares its dependencies; the working set additionally needs
`tf-keras`, `onnx_graphsurgeon`, `sng4onnx`, `simple_onnx_processing_tools`,
`psutil` and `ai_edge_litert`. Installing the export toolchain did **not**
downgrade numpy, pandas, OpenCV or torch — the P1 data suite still passes 20/20.
