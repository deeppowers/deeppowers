#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace deeppowers {
namespace kernels {

// Constants
constexpr int BLOCK_SIZE = 256;
constexpr float INT8_MAX = 127.0f;
constexpr float INT8_MIN = -128.0f;
constexpr float INT4_MAX = 7.0f;
constexpr float INT4_MIN = -8.0f;

// INT8 quantization kernel
template<typename T>
__global__ void quantize_tensor_int8_kernel(
    const T* input,
    int8_t* output,
    float* scales,
    int8_t* zero_points,
    size_t num_elements,
    size_t elements_per_channel,
    bool per_channel);

// INT4 quantization kernel
template<typename T>
__global__ void quantize_tensor_int4_kernel(
    const T* input,
    int8_t* output,
    float* scales,
    int8_t* zero_points,
    size_t num_elements,
    size_t elements_per_channel,
    bool per_channel);

// Dequantization kernel
template<typename T>
__global__ void dequantize_tensor_kernel(
    const int8_t* input,
    T* output,
    const float* scales,
    const int8_t* zero_points,
    size_t num_elements,
    size_t elements_per_channel,
    bool per_channel,
    bool is_int4);

// Calibration kernel
template<typename T>
__global__ void calibration_kernel(
    const T* input,
    float* min_vals,
    float* max_vals,
    float* running_means,
    float* running_vars,
    size_t num_elements,
    size_t elements_per_channel,
    bool per_channel);

// Helper functions
__device__ inline float clip(float x, float min_val, float max_val) {
    return fminf(fmaxf(x, min_val), max_val);
}

__device__ inline int8_t float_to_int8(float x, float scale, int8_t zero_point) {
    float quantized = x / scale + zero_point;
    return static_cast<int8_t>(clip(roundf(quantized), INT8_MIN, INT8_MAX));
}

__device__ inline int8_t float_to_int4(float x, float scale, int8_t zero_point) {
    float quantized = x / scale + zero_point;
    return static_cast<int8_t>(clip(roundf(quantized), INT4_MIN, INT4_MAX));
}

__device__ inline float int8_to_float(int8_t x, float scale, int8_t zero_point) {
    return (static_cast<float>(x) - zero_point) * scale;
}

__device__ inline float int4_to_float(int8_t packed, int index, float scale, int8_t zero_point) {
    int8_t x = (index == 0) ? (packed & 0x0F) : (packed >> 4);
    if (x >= 8) x -= 16;  // Sign extend
    return (static_cast<float>(x) - zero_point) * scale;
}

// Atomic operations for statistics
__device__ inline void atomic_min_float(float* addr, float val) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int;
    int expected;
    do {
        expected = old;
        old = atomicCAS(addr_as_int, expected,
            __float_as_int(fminf(val, __int_as_float(expected))));
    } while (old != expected);
}

__device__ inline void atomic_max_float(float* addr, float val) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int;
    int expected;
    do {
        expected = old;
        old = atomicCAS(addr_as_int, expected,
            __float_as_int(fmaxf(val, __int_as_float(expected))));
    } while (old != expected);
}

} // namespace kernels
} // namespace deeppowers 