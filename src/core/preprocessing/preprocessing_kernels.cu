#include "preprocessing_kernels.cuh"

namespace deeppowers {
namespace kernels {

// Normalization kernel implementation
template<typename T>
__global__ void normalize_kernel(
    const T* input,
    T* output,
    float mean,
    float inv_std,
    size_t num_elements) {
    
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    
    const float val = static_cast<float>(input[idx]);
    output[idx] = static_cast<T>((val - mean) * inv_std);
}

// Padding kernel implementation
template<typename T>
__global__ void padding_kernel(
    const T* input,
    T* output,
    size_t seq_length,
    size_t max_seq_length) {
    
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_seq_length) return;
    
    if (idx < seq_length) {
        output[idx] = input[idx];
    } else {
        output[idx] = static_cast<T>(0);  // Padding value
    }
}

// Masking kernel implementation
template<typename T>
__global__ void masking_kernel(
    const T* input,
    T* output,
    size_t seq_length,
    size_t max_seq_length) {
    
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_seq_length) return;
    
    // Create attention mask
    const float mask_value = (idx < seq_length) ? 1.0f : 0.0f;
    output[idx] = static_cast<T>(mask_value);
}

// Statistics collection kernel implementation
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
    size_t num_bins) {
    
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    
    const float val = static_cast<float>(input[idx]);
    
    // Update min/max values
    atomic_min_float(min_vals, val);
    atomic_max_float(max_vals, val);
    
    // Update running mean and variance
    atomic_add_double(means, val);
    atomic_add_double(variances, val * val);
    
    // Update histogram
    const int bin = find_bin(val, bin_edges, num_bins);
    if (bin >= 0 && bin < num_bins) {
        atomicAdd(&histogram[bin], 1);
    }
}

// Explicit template instantiations
template __global__ void normalize_kernel<float>(
    const float* input,
    float* output,
    float mean,
    float inv_std,
    size_t num_elements);

template __global__ void normalize_kernel<half>(
    const half* input,
    half* output,
    float mean,
    float inv_std,
    size_t num_elements);

template __global__ void padding_kernel<float>(
    const float* input,
    float* output,
    size_t seq_length,
    size_t max_seq_length);

template __global__ void padding_kernel<half>(
    const half* input,
    half* output,
    size_t seq_length,
    size_t max_seq_length);

template __global__ void masking_kernel<float>(
    const float* input,
    float* output,
    size_t seq_length,
    size_t max_seq_length);

template __global__ void masking_kernel<half>(
    const half* input,
    half* output,
    size_t seq_length,
    size_t max_seq_length);

template __global__ void statistics_kernel<float>(
    const float* input,
    float* min_vals,
    float* max_vals,
    double* means,
    double* variances,
    uint32_t* histogram,
    const float* bin_edges,
    size_t num_elements,
    size_t num_bins);

template __global__ void statistics_kernel<half>(
    const half* input,
    float* min_vals,
    float* max_vals,
    double* means,
    double* variances,
    uint32_t* histogram,
    const float* bin_edges,
    size_t num_elements,
    size_t num_bins);

} // namespace kernels
} // namespace deeppowers