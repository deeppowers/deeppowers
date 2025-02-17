#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace deeppowers {
namespace kernels {

// Constants
constexpr int BLOCK_SIZE = 256;

// Dequantization kernel
template<typename T>
__global__ void dequantize_kernel(
    const int8_t* input,
    T* output,
    const float* scales,
    const int8_t* zero_points,
    size_t num_elements,
    size_t elements_per_channel,
    bool per_channel,
    bool is_int4);

// Error computation kernel
template<typename T>
__global__ void compute_error_kernel(
    const T* original,
    const T* quantized,
    float* error_output,
    size_t num_elements);

// Statistics computation kernel
template<typename T>
__global__ void compute_statistics_kernel(
    const T* input,
    float* min_vals,
    float* max_vals,
    double* means,
    double* variances,
    uint32_t* histogram,
    const float* bin_edges,
    size_t num_elements,
    size_t num_bins);

// Validation kernel
template<typename T>
__global__ void validate_values_kernel(
    const T* input,
    uint32_t* invalid_count,
    size_t num_elements);

// Helper functions
__device__ inline float dequantize_value(
    int8_t quantized,
    float scale,
    int8_t zero_point,
    bool is_int4) {
    
    if (is_int4) {
        // Extract 4-bit value and sign extend
        int8_t val = (quantized & 0x0F);
        if (val >= 8) val -= 16;
        return (static_cast<float>(val) - zero_point) * scale;
    } else {
        return (static_cast<float>(quantized) - zero_point) * scale;
    }
}

__device__ inline float compute_absolute_error(float a, float b) {
    return fabsf(a - b);
}

__device__ inline float compute_relative_error(float a, float b) {
    if (fabsf(a) < 1e-6f) return 0.0f;
    return fabsf(a - b) / fabsf(a);
}

__device__ inline bool is_valid_value(float val) {
    return !isnan(val) && !isinf(val);
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

__device__ inline void atomic_add_double(double* addr, double val) {
    unsigned long long* addr_as_ull = (unsigned long long*)addr;
    unsigned long long old = *addr_as_ull;
    unsigned long long assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_ull, assumed,
            __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
}

} // namespace kernels
} // namespace deeppowers 