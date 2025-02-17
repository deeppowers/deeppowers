#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "gpt_kernels.hpp"

namespace deeppowers {
namespace kernels {

// Constants definition
constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;

// Utility functions
__device__ inline float gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coef = 0.044715f;
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * (x + coef * x3)));
}

// LayerNorm kernel
template<typename T>
__global__ void layer_norm_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ output,
    const int batch_size,
    const int hidden_size,
    const float eps = 1e-5) {
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    
    __shared__ float s_mean;
    __shared__ float s_variance;
    
    // Calculate mean
    float local_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += BLOCK_SIZE) {
        local_sum += static_cast<float>(input[bid * hidden_size + i]);
    }
    
    // Reduction sum
    __shared__ float s_partial_sum[BLOCK_SIZE];
    s_partial_sum[tid] = local_sum;
    __syncthreads();
    
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_partial_sum[tid] += s_partial_sum[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        s_mean = s_partial_sum[0] / hidden_size;
    }
    __syncthreads();
    
    // Calculate variance
    local_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += BLOCK_SIZE) {
        float diff = static_cast<float>(input[bid * hidden_size + i]) - s_mean;
        local_sum += diff * diff;
    }
    
    s_partial_sum[tid] = local_sum;
    __syncthreads();
    
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_partial_sum[tid] += s_partial_sum[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        s_variance = s_partial_sum[0] / hidden_size;
    }
    __syncthreads();
    
    // Normalize and apply scale and bias
    for (int i = tid; i < hidden_size; i += BLOCK_SIZE) {
        const int idx = bid * hidden_size + i;
        float val = static_cast<float>(input[idx]);
        val = (val - s_mean) / sqrtf(s_variance + eps);
        val = val * static_cast<float>(weight[i]) + static_cast<float>(bias[i]);
        output[idx] = static_cast<T>(val);
    }
}

// Flash Attention kernel
template<typename T>
__global__ void flash_attention_kernel(
    const T* __restrict__ query,
    const T* __restrict__ key,
    const T* __restrict__ value,
    const T* __restrict__ mask,
    T* __restrict__ output,
    const int batch_size,
    const int num_heads,
    const int seq_length,
    const int head_dim) {
    
    // Each block processes one attention head
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int seq_idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    if (seq_idx >= seq_length) return;
    
    // Calculate base index
    const int batch_offset = batch_idx * num_heads * seq_length * head_dim;
    const int head_offset = head_idx * seq_length * head_dim;
    const int seq_offset = seq_idx * head_dim;
    const int base_idx = batch_offset + head_offset + seq_offset;
    
    // Load query vector to shared memory
    __shared__ float s_query[BLOCK_SIZE][64];  // Assume head_dim <= 64
    for (int i = 0; i < head_dim; ++i) {
        s_query[threadIdx.x][i] = static_cast<float>(query[base_idx + i]);
    }
    __syncthreads();
    
    // Calculate attention scores
    float max_score = -INFINITY;
    float scores[BLOCK_SIZE];
    
    for (int block_start = 0; block_start < seq_length; block_start += BLOCK_SIZE) {
        const int key_idx = batch_offset + head_offset + block_start * head_dim;
        
        // Load key block to shared memory
        __shared__ float s_key[BLOCK_SIZE][64];
        if (block_start + threadIdx.x < seq_length) {
            for (int i = 0; i < head_dim; ++i) {
                s_key[threadIdx.x][i] = static_cast<float>(key[key_idx + threadIdx.x * head_dim + i]);
            }
        }
        __syncthreads();
        
        // Calculate current block's attention scores
        const int valid_len = min(BLOCK_SIZE, seq_length - block_start);
        for (int i = 0; i < valid_len; ++i) {
            float score = 0.0f;
            for (int j = 0; j < head_dim; ++j) {
                score += s_query[threadIdx.x][j] * s_key[i][j];
            }
            score /= sqrtf(static_cast<float>(head_dim));
            
            // Apply mask
            if (mask) {
                score += static_cast<float>(mask[batch_idx * seq_length * seq_length + 
                                               seq_idx * seq_length + 
                                               block_start + i]);
            }
            
            scores[i] = score;
            max_score = max(max_score, score);
        }
        __syncthreads();
    }
    
    // Calculate softmax
    float sum_exp = 0.0f;
    for (int i = 0; i < seq_length; ++i) {
        scores[i] = expf(scores[i] - max_score);
        sum_exp += scores[i];
    }
    
    for (int i = 0; i < seq_length; ++i) {
        scores[i] /= sum_exp;
    }
    
    // Calculate weighted sum
    float output_vec[64] = {0.0f};  // Assume head_dim <= 64
    for (int i = 0; i < seq_length; ++i) {
        const int value_idx = batch_offset + head_offset + i * head_dim;
        for (int j = 0; j < head_dim; ++j) {
            output_vec[j] += scores[i] * static_cast<float>(value[value_idx + j]);
        }
    }
    
    // Write back output
    for (int i = 0; i < head_dim; ++i) {
        output[base_idx + i] = static_cast<T>(output_vec[i]);
    }
}

// FFN kernel
template<typename T>
__global__ void ffn_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight1,
    const T* __restrict__ bias1,
    const T* __restrict__ weight2,
    const T* __restrict__ bias2,
    T* __restrict__ output,
    const int batch_size,
    const int seq_length,
    const int hidden_size,
    const int intermediate_size) {
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int seq_id = blockIdx.y;
    const int batch_id = blockIdx.z;
    
    // Calculate base index
    const int base_idx = (batch_id * seq_length + seq_id) * hidden_size;
    
    // First linear layer + GELU
    float intermediate[4096];  // Assume intermediate_size <= 4096
    for (int i = tid; i < intermediate_size; i += BLOCK_SIZE) {
        float sum = 0.0f;
        for (int j = 0; j < hidden_size; ++j) {
            sum += static_cast<float>(input[base_idx + j]) * 
                   static_cast<float>(weight1[j * intermediate_size + i]);
        }
        sum += static_cast<float>(bias1[i]);
        intermediate[i] = gelu(sum);
    }
    __syncthreads();
    
    // Second linear layer
    for (int i = tid; i < hidden_size; i += BLOCK_SIZE) {
        float sum = 0.0f;
        for (int j = 0; j < intermediate_size; ++j) {
            sum += intermediate[j] * static_cast<float>(weight2[j * hidden_size + i]);
        }
        sum += static_cast<float>(bias2[i]);
        output[base_idx + i] = static_cast<T>(sum);
    }
}

// Rotation position encoding kernel
template<typename T>
__global__ void rotary_embedding_kernel(
    T* __restrict__ query,
    T* __restrict__ key,
    const int batch_size,
    const int num_heads,
    const int seq_length,
    const int head_dim,
    const float base = 10000.0f) {
    
    const int tid = threadIdx.x;
    const int head_idx = blockIdx.y;
    const int seq_idx = blockIdx.x;
    const int batch_idx = blockIdx.z;
    
    if (tid >= head_dim / 2) return;
    
    // Calculate base index
    const int batch_offset = batch_idx * num_heads * seq_length * head_dim;
    const int head_offset = head_idx * seq_length * head_dim;
    const int seq_offset = seq_idx * head_dim;
    const int base_idx = batch_offset + head_offset + seq_offset;
    
    // Calculate rotation angle
    const float theta = powf(base, -2.0f * tid / head_dim);
    const float sin_pos = sinf(seq_idx * theta);
    const float cos_pos = cosf(seq_idx * theta);
    
    // Apply rotation
    const int dim_idx = tid * 2;
    const int q_idx = base_idx + dim_idx;
    const int k_idx = base_idx + dim_idx;
    
    // Process query
    float q0 = static_cast<float>(query[q_idx]);
    float q1 = static_cast<float>(query[q_idx + 1]);
    query[q_idx] = static_cast<T>(q0 * cos_pos - q1 * sin_pos);
    query[q_idx + 1] = static_cast<T>(q0 * sin_pos + q1 * cos_pos);
    
    // Process key
    float k0 = static_cast<float>(key[k_idx]);
    float k1 = static_cast<float>(key[k_idx + 1]);
    key[k_idx] = static_cast<T>(k0 * cos_pos - k1 * sin_pos);
    key[k_idx + 1] = static_cast<T>(k0 * sin_pos + k1 * cos_pos);
}

// Embedding lookup kernel
template<typename T>
__global__ void embedding_lookup_kernel(
    const int32_t* __restrict__ input_ids,
    const T* __restrict__ token_embedding,
    const T* __restrict__ position_embedding,
    T* __restrict__ output,
    const int batch_size,
    const int seq_length,
    const int hidden_size) {
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int seq_id = blockIdx.y;
    const int batch_id = blockIdx.z;
    
    if (tid >= hidden_size) return;
    
    // Calculate base index
    const int token_id = input_ids[batch_id * seq_length + seq_id];
    const int output_idx = (batch_id * seq_length + seq_id) * hidden_size + tid;
    
    // Get token embedding and position encoding
    const T token_vec = token_embedding[token_id * hidden_size + tid];
    const T pos_vec = position_embedding[seq_id * hidden_size + tid];
    
    // Merge embeddings
    output[output_idx] = token_vec + pos_vec;
}

// QKV transformation kernel
template<typename T>
__global__ void qkv_transform_kernel(
    const T* __restrict__ input,
    const T* __restrict__ q_weight,
    const T* __restrict__ k_weight,
    const T* __restrict__ v_weight,
    const T* __restrict__ q_bias,
    const T* __restrict__ k_bias,
    const T* __restrict__ v_bias,
    T* __restrict__ query,
    T* __restrict__ key,
    T* __restrict__ value,
    const int batch_size,
    const int seq_length,
    const int hidden_size,
    const int num_heads,
    const int head_dim) {
    
    const int tid = threadIdx.x;
    const int head_idx = blockIdx.x;
    const int seq_id = blockIdx.y;
    const int batch_id = blockIdx.z;
    
    if (tid >= head_dim) return;
    
    // Calculate base index
    const int input_idx = (batch_id * seq_length + seq_id) * hidden_size;
    const int head_offset = head_idx * head_dim;
    const int output_idx = ((batch_id * num_heads + head_idx) * seq_length + seq_id) * head_dim + tid;
    
    // Calculate Q, K, V
    float q_val = 0.0f, k_val = 0.0f, v_val = 0.0f;
    
    #pragma unroll
    for (int i = 0; i < hidden_size; ++i) {
        const float input_val = static_cast<float>(input[input_idx + i]);
        q_val += input_val * static_cast<float>(q_weight[i * hidden_size + head_offset + tid]);
        k_val += input_val * static_cast<float>(k_weight[i * hidden_size + head_offset + tid]);
        v_val += input_val * static_cast<float>(v_weight[i * hidden_size + head_offset + tid]);
    }
    
    // Add bias and write back
    query[output_idx] = static_cast<T>(q_val + static_cast<float>(q_bias[head_offset + tid]));
    key[output_idx] = static_cast<T>(k_val + static_cast<float>(k_bias[head_offset + tid]));
    value[output_idx] = static_cast<T>(v_val + static_cast<float>(v_bias[head_offset + tid]));
}

// Residual addition kernel
template<typename T>
__global__ void residual_add_kernel(
    const T* __restrict__ input,
    const T* __restrict__ residual,
    T* __restrict__ output,
    const int batch_size,
    const int seq_length,
    const int hidden_size) {
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int seq_id = blockIdx.y;
    const int batch_id = blockIdx.z;
    
    if (tid >= hidden_size) return;
    
    const int idx = (batch_id * seq_length + seq_id) * hidden_size + tid;
    output[idx] = input[idx] + residual[idx];
}

// Logits calculation kernel
template<typename T>
__global__ void compute_logits_kernel(
    const T* __restrict__ hidden_states,
    const T* __restrict__ lm_head_weight,
    const T* __restrict__ lm_head_bias,
    T* __restrict__ logits,
    const int batch_size,
    const int seq_length,
    const int hidden_size,
    const int vocab_size) {
    
    const int tid = threadIdx.x;
    const int vocab_chunk = blockIdx.x;
    const int seq_id = blockIdx.y;
    const int batch_id = blockIdx.z;
    
    // Each block processes a part of the vocabulary
    const int vocab_start = vocab_chunk * BLOCK_SIZE;
    const int vocab_idx = vocab_start + tid;
    
    if (vocab_idx >= vocab_size) return;
    
    // Calculate base index
    const int hidden_idx = (batch_id * seq_length + seq_id) * hidden_size;
    const int output_idx = (batch_id * seq_length + seq_id) * vocab_size + vocab_idx;
    
    // Calculate logit
    float logit = 0.0f;
    
    #pragma unroll
    for (int i = 0; i < hidden_size; ++i) {
        logit += static_cast<float>(hidden_states[hidden_idx + i]) * 
                static_cast<float>(lm_head_weight[vocab_idx * hidden_size + i]);
    }
    
    logit += static_cast<float>(lm_head_bias[vocab_idx]);
    logits[output_idx] = static_cast<T>(logit);
}

// Quantization related kernel
namespace {

// Helper function: Calculate quantization parameters
__device__ inline void calculate_quant_params(float min_val, float max_val, float& scale, int8_t& zero_point) {
    const float epsilon = 1e-8f;
    max_val = max(abs(max_val), abs(min_val));
    scale = max_val / 127.0f;
    scale = max(scale, epsilon);
    zero_point = 0;
}

// Helper function: Quantize a single value to INT8
__device__ inline int8_t quantize_to_int8(float val, float scale, int8_t zero_point) {
    float scaled_val = val / scale;
    return static_cast<int8_t>(round(scaled_val) + zero_point);
}

// Helper function: Quantize a single value to INT4
__device__ inline int8_t quantize_to_int4(float val, float scale, int8_t zero_point) {
    float scaled_val = val / scale;
    return static_cast<int8_t>(min(max(round(scaled_val) + zero_point, -8.0f), 7.0f));
}

} // anonymous namespace

// INT8 quantization kernel
__global__ void quantize_tensor_int8_kernel(
    const float* __restrict__ input,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    int8_t* __restrict__ zero_points,
    const int num_elements,
    const int elements_per_channel,
    const bool per_channel) {
    
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int channel_id = per_channel ? tid / elements_per_channel : 0;
    
    if (tid >= num_elements) return;
    
    if (per_channel) {
        // Each channel calculates quantization parameters separately
        if (tid % elements_per_channel == 0) {
            float min_val = FLT_MAX;
            float max_val = -FLT_MAX;
            
            for (int i = 0; i < elements_per_channel; ++i) {
                float val = input[channel_id * elements_per_channel + i];
                min_val = min(min_val, val);
                max_val = max(max_val, val);
            }
            
            calculate_quant_params(min_val, max_val, scales[channel_id], zero_points[channel_id]);
        }
        __syncthreads();
        
        output[tid] = quantize_to_int8(input[tid], scales[channel_id], zero_points[channel_id]);
    } else {
        // Global quantization parameters
        if (tid == 0) {
            float min_val = FLT_MAX;
            float max_val = -FLT_MAX;
            
            for (int i = 0; i < num_elements; ++i) {
                float val = input[i];
                min_val = min(min_val, val);
                max_val = max(max_val, val);
            }
            
            calculate_quant_params(min_val, max_val, scales[0], zero_points[0]);
        }
        __syncthreads();
        
        output[tid] = quantize_to_int8(input[tid], scales[0], zero_points[0]);
    }
}

// INT4 quantization kernel
__global__ void quantize_tensor_int4_kernel(
    const float* __restrict__ input,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    int8_t* __restrict__ zero_points,
    const int num_elements,
    const int elements_per_channel,
    const bool per_channel) {
    
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int channel_id = per_channel ? tid / elements_per_channel : 0;
    
    if (tid >= num_elements) return;
    
    if (per_channel) {
        // Each channel calculates quantization parameters separately
        if (tid % elements_per_channel == 0) {
            float min_val = FLT_MAX;
            float max_val = -FLT_MAX;
            
            for (int i = 0; i < elements_per_channel; ++i) {
                float val = input[channel_id * elements_per_channel + i];
                min_val = min(min_val, val);
                max_val = max(max_val, val);
            }
            
            calculate_quant_params(min_val, max_val, scales[channel_id], zero_points[channel_id]);
        }
        __syncthreads();
        
        // Two INT4 values are packed into one INT8
        if (tid % 2 == 0) {
            int8_t low = quantize_to_int4(input[tid], scales[channel_id], zero_points[channel_id]);
            int8_t high = tid + 1 < num_elements ? 
                quantize_to_int4(input[tid + 1], scales[channel_id], zero_points[channel_id]) : 0;
            output[tid / 2] = (high << 4) | (low & 0x0F);
        }
    } else {
        // Global quantization parameters
        if (tid == 0) {
            float min_val = FLT_MAX;
            float max_val = -FLT_MAX;
            
            for (int i = 0; i < num_elements; ++i) {
                float val = input[i];
                min_val = min(min_val, val);
                max_val = max(max_val, val);
            }
            
            calculate_quant_params(min_val, max_val, scales[0], zero_points[0]);
        }
        __syncthreads();
        
        // Two INT4 values are packed into one INT8
        if (tid % 2 == 0) {
            int8_t low = quantize_to_int4(input[tid], scales[0], zero_points[0]);
            int8_t high = tid + 1 < num_elements ? 
                quantize_to_int4(input[tid + 1], scales[0], zero_points[0]) : 0;
            output[tid / 2] = (high << 4) | (low & 0x0F);
        }
    }
}

// Dequantization kernel
__global__ void dequantize_tensor_kernel(
    const int8_t* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ scales,
    const int8_t* __restrict__ zero_points,
    const int num_elements,
    const int elements_per_channel,
    const bool per_channel,
    const bool is_int4) {
    
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int channel_id = per_channel ? tid / elements_per_channel : 0;
    
    if (tid >= num_elements) return;
    
    float scale = scales[channel_id];
    int8_t zero_point = zero_points[channel_id];
    
    if (is_int4) {
        // Unpack two INT4 values from INT8
        int8_t packed = input[tid / 2];
        int8_t val;
        if (tid % 2 == 0) {
            val = packed & 0x0F;
        } else {
            val = (packed >> 4) & 0x0F;
        }
        
        // Handle sign bit
        if (val & 0x8) {
            val |= 0xF0;
        }
        
        output[tid] = (static_cast<float>(val) - zero_point) * scale;
    } else {
        output[tid] = (static_cast<float>(input[tid]) - zero_point) * scale;
    }
}

// Calibration kernel
__global__ void calibration_kernel(
    const float* __restrict__ input,
    float* __restrict__ min_vals,
    float* __restrict__ max_vals,
    float* __restrict__ running_means,
    float* __restrict__ running_vars,
    const int num_elements,
    const int elements_per_channel,
    const bool per_channel) {
    
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int channel_id = per_channel ? tid / elements_per_channel : 0;
    
    if (tid >= num_elements) return;
    
    if (per_channel) {
        // Each channel calculates statistical information separately
        if (tid % elements_per_channel == 0) {
            float sum = 0.0f;
            float sq_sum = 0.0f;
            float min_val = FLT_MAX;
            float max_val = -FLT_MAX;
            
            for (int i = 0; i < elements_per_channel; ++i) {
                float val = input[channel_id * elements_per_channel + i];
                sum += val;
                sq_sum += val * val;
                min_val = min(min_val, val);
                max_val = max(max_val, val);
            }
            
            float mean = sum / elements_per_channel;
            float variance = (sq_sum / elements_per_channel) - (mean * mean);
            
            // Update statistical information
            atomicMin((int*)&min_vals[channel_id], __float_as_int(min_val));
            atomicMax((int*)&max_vals[channel_id], __float_as_int(max_val));
            running_means[channel_id] = mean;
            running_vars[channel_id] = variance;
        }
    } else {
        // Global statistical information
        if (tid == 0) {
            float sum = 0.0f;
            float sq_sum = 0.0f;
            float min_val = FLT_MAX;
            float max_val = -FLT_MAX;
            
            for (int i = 0; i < num_elements; ++i) {
                float val = input[i];
                sum += val;
                sq_sum += val * val;
                min_val = min(min_val, val);
                max_val = max(max_val, val);
            }
            
            float mean = sum / num_elements;
            float variance = (sq_sum / num_elements) - (mean * mean);
            
            // Update statistical information
            min_vals[0] = min_val;
            max_vals[0] = max_val;
            running_means[0] = mean;
            running_vars[0] = variance;
        }
    }
}

// Instantiate templates
template __global__ void layer_norm_kernel<float>(const float*, const float*, const float*, float*, const int, const int, const float);
template __global__ void layer_norm_kernel<half>(const half*, const half*, const half*, half*, const int, const int, const float);

template __global__ void flash_attention_kernel<float>(const float*, const float*, const float*, const float*, float*, const int, const int, const int, const int);
template __global__ void flash_attention_kernel<half>(const half*, const half*, const half*, const half*, half*, const int, const int, const int, const int);

template __global__ void ffn_kernel<float>(const float*, const float*, const float*, const float*, const float*, float*, const int, const int, const int, const int);
template __global__ void ffn_kernel<half>(const half*, const half*, const half*, const half*, const half*, half*, const int, const int, const int, const int);

template __global__ void rotary_embedding_kernel<float>(float*, float*, const int, const int, const int, const int, const float);
template __global__ void rotary_embedding_kernel<half>(half*, half*, const int, const int, const int, const int, const float);

template __global__ void embedding_lookup_kernel<float>(const int32_t*, const float*, const float*, float*, const int, const int, const int);
template __global__ void embedding_lookup_kernel<half>(const int32_t*, const half*, const half*, half*, const int, const int, const int);

template __global__ void qkv_transform_kernel<float>(const float*, const float*, const float*, const float*, const float*, const float*, const float*, float*, float*, float*, const int, const int, const int, const int, const int);
template __global__ void qkv_transform_kernel<half>(const half*, const half*, const half*, const half*, const half*, const half*, const half*, half*, half*, half*, const int, const int, const int, const int, const int);

template __global__ void residual_add_kernel<float>(const float*, const float*, float*, const int, const int, const int);
template __global__ void residual_add_kernel<half>(const half*, const half*, half*, const int, const int, const int);

template __global__ void compute_logits_kernel<float>(const float*, const float*, const float*, float*, const int, const int, const int, const int);
template __global__ void compute_logits_kernel<half>(const half*, const half*, const half*, half*, const int, const int, const int, const int);

template __global__ void quantize_tensor_int8_kernel(const float*, int8_t*, float*, int8_t*, const int, const int, const bool);
template __global__ void quantize_tensor_int4_kernel(const float*, int8_t*, float*, int8_t*, const int, const int, const bool);
template __global__ void dequantize_tensor_kernel(const int8_t*, float*, const float*, const int8_t*, const int, const int, const bool, const bool);
template __global__ void calibration_kernel(const float*, float*, float*, float*, float*, const int, const int, const bool);

} // namespace kernels
} // namespace deeppowers