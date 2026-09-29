#ifndef LEAF_DISEASE_MODEL_H
#define LEAF_DISEASE_MODEL_H

#ifdef __cplusplus
extern "C" {
#endif

// Quantized TFLite model for leaf disease detection
// Size: 671256 bytes
// Input shape: 128x128x3 INT8
// Classes: 39

extern const unsigned int g_leaf_disease_model_len;
#ifdef __cplusplus
alignas(16) extern const unsigned char g_leaf_disease_model[];
#else
__attribute__((aligned(16))) extern const unsigned char g_leaf_disease_model[];
#endif

#ifdef __cplusplus
}
#endif

#endif // LEAF_DISEASE_MODEL_H
