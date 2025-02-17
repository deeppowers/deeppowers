#include "quantization_kernels.cuh"

namespace deeppowers {
namespace kernels {

// INT8 quantization kernel implementation
template<typename T>
__global__ void quantize_tensor_int8_kernel(
    const T* input,
    int8_t* output,
    float* scales,
    int8_t* zero_points,
    size_t num_elements,
    size_t elements_per_channel,
    bool per_channel) {
    
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    
    const size_t channel = per_channel ? (idx / elements_per_channel) : 0;
    const float scale = scales[channel];
    const int8_t zero_point = zero_points[channel];
    
    output[idx] = float_to_int8(static_cast<float>(input[idx]), scale, zero_point);
}

// INT4 quantization kernel implementation
template<typename T>
__global__ void quantize_tensor_int4_kernel(
    const T* input,
    int8_t* output,
    float* scales,
    int8_t* zero_points,
    size_t num_elements,
    size_t elements_per_channel,
    bool per_channel) {
    
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (num_elements + 1) / 2) return;
    
    // Process two elements at a time
    const size_t input_idx1 = idx * 2;
    const size_t input_idx2 = min(input_idx1 + 1, num_elements - 1);
    
    const size_t channel1 = per_channel ? (input_idx1 / elements_per_channel) : 0;
    const size_t channel2 = per_channel ? (input_idx2 / elements_per_channel) : 0;
    
    // Quantize first element
    int8_t low = float_to_int4(
        static_cast<float>(input[input_idx1]),
        scales[channel1],
        zero_points[channel1]);
    
    // Quantize second element
    int8_t high = float_to_int4(
        static_cast<float>(input[input_idx2]),
        scales[channel2],
        zero_points[channel2]);
    
    // Pack two INT4 values into one INT8
    output[idx] = (high << 4) | (low & 0x0F);
}

// Dequantization kernel implementation
template<typename T>
__global__ void dequantize_tensor_kernel(
    const int8_t* input,
    T* output,
    const float* scales,
    const int8_t* zero_points,
    size_t num_elements,
    size_t elements_per_channel,
    bool per_channel,
    bool is_int4) {
    
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    
    const size_t channel = per_channel ? (idx / elements_per_channel) : 0;
    const float scale = scales[channel];
    const int8_t zero_point = zero_points[channel];
    
    if (is_int4) {
        // Handle INT4 dequantization
        const size_t packed_idx = idx / 2;
        const int sub_idx = idx % 2;
        const int8_t packed = input[packed_idx];
        output[idx] = static_cast<T>(
            int4_to_float(packed, sub_idx, scale, zero_point));
    } else {
        // Handle INT8 dequantization
        output[idx] = static_cast<T>(
            int8_to_float(input[idx], scale, zero_point));
    }
}

// Calibration kernel implementation
template<typename T>
__global__ void calibration_kernel(
    const T* input,
    float* min_vals,
    float* max_vals,
    float* running_means,
    float* running_vars,
    size_t num_elements,
    size_t elements_per_channel,
    bool per_channel) {
    
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    
    const float val = static_cast<float>(input[idx]);
    const size_t channel = per_channel ? (idx / elements_per_channel) : 0;
    
    // Update min/max values
    atomic_min_float(&min_vals[channel], val);
    atomic_max_float(&max_vals[channel], val);
    
    // Update running mean and variance
    // Note: This is a simplified implementation
    // For production use, consider using Welford's online algorithm
    atomicAdd(&running_means[channel], val);
    atomicAdd(&running_vars[channel], val * val);
}

// Explicit template instantiations
template __global__ void quantize_tensor_int8_kernel<float>(
    const float* input,
    int8_t* output,
    float* scales,
    int8_t* zero_points,
    size_t num_elements,
    size_t elements_per_channel,
    bool per_channel);

template __global__ void quantize_tensor_int8_kernel<half>(
    const half* input,
    int8_t* output,
    float* scales,
    int8_t* zero_points,
    size_t num_elements,
    size_t elements_per_channel,
    bool per_channel);

template __global__ void quantize_tensor_int4_kernel<float>(
    const float* input,
    int8_t* output,
    float* scales,
    int8_t* zero_points,
    size_t num_elements,
    size_t elements_per_channel,
    bool per_channel);

template __global__ void quantize_tensor_int4_kernel<half>(
    const half* input,
    int8_t* output,
    float* scales,
    int8_t* zero_points,
    size_t num_elements,
    size_t elements_per_channel,
    bool per_channel);

template __global__ void dequantize_tensor_kernel<float>(
    const int8_t* input,
    float* output,
    const float* scales,
    const int8_t* zero_points,
    size_t num_elements,
    size_t elements_per_channel,
    bool per_channel,
    bool is_int4);

template __global__ void dequantize_tensor_kernel<half>(
    const int8_t* input,
    half* output,
    const float* scales,
    const int8_t* zero_points,
    size_t num_elements,
    size_t elements_per_channel,
    bool per_channel,
    bool is_int4);

template __global__ void calibration_kernel<float>(
    const float* input,
    float* min_vals,
    float* max_vals,
    float* running_means,
    float* running_vars,
    size_t num_elements,
    size_t elements_per_channel,
    bool per_channel);

template __global__ void calibration_kernel<half>(
    const half* input,
    float* min_vals,
    float* max_vals,
    float* running_means,
    float* running_vars,
    size_t num_elements,
    size_t elements_per_channel,
    bool per_channel);

} // namespace kernels
} // namespace deeppowers 