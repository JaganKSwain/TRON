#include <stdint.h>
#include "hal_data.h" // Pulls in all FSP drivers, including the Ethos-U55
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Ensure this matches your actual exported model header name
#include "leaf_disease_model.h"

extern "C" {
#include <tk/tkernel.h>
#include <tm/tmonitor.h>
}

// Memory arena for tensor calculations
constexpr int kTensorArenaSize = 256 * 1024;
static uint8_t tensor_arena[kTensorArenaSize] __attribute__((aligned(16)));

static const tflite::Model* model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input_tensor = nullptr;
static TfLiteTensor* output_tensor = nullptr;

extern "C" bool ai_init(void) {
    // 1. Initialize Ethos-U55 using the official FSP API
    fsp_err_t err = RM_ETHOSU_Open(&g_rm_ethosu0_ctrl, &g_rm_ethosu0_cfg);
    if (err != FSP_SUCCESS) {
        tm_printf("[AI Error] Failed to open Ethos-U55 driver!\n");
        return false;
    }

    // 2. Map the Vela-compiled model array
    model = tflite::GetModel(g_leaf_disease_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        tm_printf("[AI Error] Model schema mismatch!\n");
        return false;
    }

    // 3. Register Ethos-U operator
    static tflite::MicroMutableOpResolver<2> resolver;
    resolver.AddEthosU();
    resolver.AddSoftmax();

    // 4. Build the interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        tm_printf("[AI Error] AllocateTensors failed!\n");
        return false;
    }

    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);
    tm_printf("[AI] Leaf Disease Model initialized on Ethos-U55.\n");
    return true;
}

extern "C" int ai_run_inference(void) {
    if (!interpreter || !input_tensor) return -1;

    // TODO: Connect official EK-RA8P1 camera buffer here later

    if (interpreter->Invoke() != kTfLiteOk) {
        tm_printf("[AI Error] Inference failed!\n");
        return -1;
    }

    int8_t* output_data = output_tensor->data.int8;
    int best_class = 0;
    int8_t max_score = -128;

    for (int i = 0; i < output_tensor->bytes; i++) {
        if (output_data[i] > max_score) {
            max_score = output_data[i];
            best_class = i;
        }
    }
    return best_class;
}
