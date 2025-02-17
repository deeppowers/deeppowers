#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace deeppowers {
namespace kernels {

// Constants
constexpr int BLOCK_SIZE = 256;

// Normalization kernel
template<typename T>
__global__ void normalize_kernel(
    const T* input,
    T* output,
    float mean,
    float inv_std,
    size_t num_elements);

// Padding kernel
template<typename T>
__global__ void padding_kernel(
    const T* input,
    T* output,
    size_t seq_length,
    size_t max_seq_length);

// Masking kernel
template<typename T>
__global__ void masking_kernel(
    const T* input,
    T* output,
    size_t seq_length,
    size_t max_seq_length);

// Statistics collection kernel
template<typename T>
__global__ void statistics_kernel(
    const T* input,
    float* min_vals,
    float* max_vals,
    double* means,
    double* variances,
    uint32_t* histogram,
    const float* bin_edges,
    size_t num_elements,
    size_t num_bins);

// Helper functions
__device__ inline float clip(float x, float min_val, float max_val) {
    return fminf(fmaxf(x, min_val), max_val);
}

__device__ inline int find_bin(float val, const float* bin_edges, size_t num_bins) {
    for (size_t i = 0; i < num_bins; ++i) {
        if (val >= bin_edges[i] && val < bin_edges[i + 1]) {
            return i;
        }
    }
    return num_bins - 1;
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