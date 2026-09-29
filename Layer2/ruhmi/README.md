# AgroBot RUHMI / Renesas RA8P1 Deployment Guide

This directory contains the firmware deployment assets for running AgroBot on the **Renesas RA8P1 Microcontroller** with the **Arm Ethos-U55 NPU**.

---

## Model Specifications
- **Architecture**: MobileNetV2 (`width_mult=0.5`) with 100% `ReLU6` activations.
- **Quantization**: Fully integer INT8 (per-channel weights, per-tensor activations).
- **Input Dimension**: `160 x 160 x 3` (`int8`, NHWC order).
- **Output**: 15 classes (`int8` raw logits).
- **SRAM Usage**: `402.1 KiB` (fits in 2 MB SRAM).
- **Flash/MRAM Size**: `785.6 KiB` (fits in 1 MB MRAM / 8 MB Flash).
- **NPU Offload**: **100.0%** (65/65 operators executed in hardware on Ethos-U55).
- **Estimated Inference Latency**: **3.82 ms (~261 FPS)** @ 500 MHz.

---

## Deployment Steps in e2studio / FSP

### 1. Include Model Data
Include `model_data.h` into your e2studio / Keil project:
```c
#include "model_data.h"
#include "ethosu_driver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
```

### 2. Initialize Tensor Arena
Allocate a 512 KiB SRAM buffer in non-cached DTCM/SRAM:
```c
#define TENSOR_ARENA_SIZE (512 * 1024)
static uint8_t tensor_arena[TENSOR_ARENA_SIZE] __attribute__((aligned(16)));
```

### 3. Setup Ethos-U55 Driver & Interpreter
```c
// 1. Initialize NPU Driver
struct ethosu_driver ethosu_drv;
ethosu_init(&ethosu_drv, (void *)ETHOSU_BASE_ADDRESS, NULL, 0, 1, 1);

// 2. Load Model from Flash/MRAM
const tflite::Model *model = tflite::GetModel(g_agrobot_model_data);

// 3. Instantiate MicroInterpreter with Ethos-U Op Resolver
static tflite::MicroMutableOpResolver<1> op_resolver;
op_resolver.AddCustom(tflite::GetString_ETHOSU(), tflite::Register_ETHOSU());

static tflite::MicroInterpreter interpreter(
    model, op_resolver, tensor_arena, TENSOR_ARENA_SIZE, nullptr
);
interpreter.AllocateTensors();
```

### 4. Camera Ingestion & Preprocessing
Input tensor expects `int8_t` values normalized with ImageNet mean/std:
$$x_{	ext{int8}} = 	ext{round}\left(rac{x_{	ext{norm}}}{S}ight) + Z$$
```c
TfLiteTensor *input = interpreter.input(0);
int8_t *input_data = input->data.int8;

// Copy 160x160x3 RGB camera frame with zero-copy DMA or memcpy
for (int i = 0; i < 160 * 160 * 3; i++) {
    input_data[i] = (int8_t)(camera_buffer[i] - 128);
}
```

### 5. Run Hardware Inference & Parse Output
```c
interpreter.Invoke();

TfLiteTensor *output = interpreter.output(0);
int8_t *logits = output->data.int8;

// Find argmax disease index
int best_class = 0;
int8_t max_logit = -128;
for (int c = 0; c < 15; c++) {
    if (logits[c] > max_logit) {
        max_logit = logits[c];
        best_class = c;
    }
}
```

---

## 15 Class Mapping
```c
static const char *const PLANT_CLASSES[15] = {
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
};
```
