#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace deeppowers {
namespace kernels {

// Constant definitions
constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;

// Matrix multiplication kernel
template<typename T>
__global__ void matmul_kernel(
    const T* A,
    const T* B,
    T* C,
    int M,
    int N,
    int K,
    T alpha = 1.0f,
    T beta = 0.0f);

// Layer normalization kernel
template<typename T>
__global__ void layernorm_kernel(
    const T* input,
    const T* gamma,
    const T* beta,
    T* output,
    int batch_size,
    int hidden_size,
    float eps = 1e-5f);

// Attention calculation kernel
template<typename T>
__global__ void attention_kernel(
    const T* query,
    const T* key,
    const T* value,
    const T* mask,
    T* output,
    int batch_size,
    int num_heads,
    int seq_length,
    int head_dim);

// Activation function kernel
template<typename T>
__global__ void activation_kernel(
    const T* input,
    T* output,
    int size,
    const char* activation_type);

// Element-wise operation kernel
template<typename T>
__global__ void elementwise_kernel(
    const T* input_a,
    const T* input_b,
    T* output,
    int size,
    const char* op_type);

// Tensor transpose kernel
template<typename T>
__global__ void transpose_kernel(
    const T* input,
    T* output,
    const int* dims,
    const int* perm,
    int rank);

// Utility functions
__device__ inline float gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coef = 0.044715f;
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * (x + coef * x3)));
}

__device__ inline float softmax(float x, float max_val, float sum) {
    return __expf(x - max_val) / sum;
}

// Atomic operations
__device__ inline void atomic_add_float(float* addr, float val) {
    atomicAdd(addr, val);
}

__device__ inline void atomic_add_half(half* addr, half val) {
    unsigned int* addr_as_uint = (unsigned int*)addr;
    unsigned int old = *addr_as_uint;
    unsigned int assumed;
    do {
        assumed = old;
        half sum = __hadd(val, __ushort_as_half(assumed));
        old = atomicCAS(addr_as_uint, assumed, __half_as_ushort(sum));
    } while (assumed != old);
}

// Shared memory management
template<typename T>
struct SharedMemory {
    __device__ inline operator T*() {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }
};

// Specialized version
template<>
struct SharedMemory<half> {
    __device__ inline operator half*() {
        extern __shared__ int __smem_half[];
        return (half*)__smem_half;
    }
};

} // namespace kernels
} // namespace deeppowers 