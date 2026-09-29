# Model B (MCU MobileNetV2-0.5) Distillation Summary

## Performance
- **Architecture**: MobileNetV2 (`width_mult=0.5`)
- **Input Resolution**: `160x160`
- **Best Validation Epoch**: 13
- **Best Validation Macro-F1**: **0.9710**
- **Test Set Accuracy**: **0.9690**
- **Test Set Macro-F1**: **0.9675**
- **Test Balanced Accuracy**: **0.9654**

## Knowledge Distillation Hyperparameters
- Teacher: Model A (`convnext_tiny` + GeM)
- Distillation Temperature: `3.0`
- KD Loss Weight ($lpha$): `0.6`
- Optimizer: AdamW, lr=0.001, epochs=15
