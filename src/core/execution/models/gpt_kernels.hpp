#pragma once

namespace deeppowers {
namespace kernels {

// LayerNorm kernel
template<typename T>
__global__ void layer_norm_kernel(
    const T* input,
    const T* weight,
    const T* bias,
    T* output,
    const int batch_size,
    const int hidden_size,
    const float eps = 1e-5);

// Flash Attention kernel
template<typename T>
__global__ void flash_attention_kernel(
    const T* query,
    const T* key,
    const T* value,
    const T* mask,
    T* output,
    const int batch_size,
    const int num_heads,
    const int seq_length,
    const int head_dim);

// FFN kernel
template<typename T>
__global__ void ffn_kernel(
    const T* input,
    const T* weight1,
    const T* bias1,
    const T* weight2,
    const T* bias2,
    T* output,
    const int batch_size,
    const int seq_length,
    const int hidden_size,
    const int intermediate_size);

// Rotary Position Embedding kernel
template<typename T>
__global__ void rotary_embedding_kernel(
    T* query,
    T* key,
    const int batch_size,
    const int num_heads,
    const int seq_length,
    const int head_dim,
    const float base = 10000.0f);

// Embedding lookup kernel
template<typename T>
__global__ void embedding_lookup_kernel(
    const int32_t* input_ids,
    const T* token_embedding,
    const T* position_embedding,
    T* output,
    const int batch_size,
    const int seq_length,
    const int hidden_size);

// QKV transformation kernel
template<typename T>
__global__ void qkv_transform_kernel(
    const T* input,
    const T* q_weight,
    const T* k_weight,
    const T* v_weight,
    const T* q_bias,
    const T* k_bias,
    const T* v_bias,
    T* query,
    T* key,
    T* value,
    const int batch_size,
    const int seq_length,
    const int hidden_size,
    const int num_heads,
    const int head_dim);

// Residual connection kernel
template<typename T>
__global__ void residual_add_kernel(
    const T* input,
    const T* residual,
    T* output,
    const int batch_size,
    const int seq_length,
    const int hidden_size);

// Compute logits kernel
template<typename T>
__global__ void compute_logits_kernel(
    const T* hidden_states,
    const T* lm_head_weight,
    const T* lm_head_bias,
    T* logits,
    const int batch_size,
    const int seq_length,
    const int hidden_size,
    const int vocab_size);

} // namespace kernels
} // namespace deeppowers 