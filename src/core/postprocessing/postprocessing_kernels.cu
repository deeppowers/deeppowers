#include "postprocessing_kernels.cuh"

namespace deeppowers {
namespace kernels {

// Dequantization kernel implementation
template<typename T>
__global__ void dequantize_kernel(
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
        const int8_t val = (sub_idx == 0) ? (packed & 0x0F) : (packed >> 4);
        output[idx] = static_cast<T>(dequantize_value(val, scale, zero_point, true));
    } else {
        // Handle INT8 dequantization
        output[idx] = static_cast<T>(dequantize_value(input[idx], scale, zero_point, false));
    }
}

// Error computation kernel implementation
template<typename T>
__global__ void compute_error_kernel(
    const T* original,
    const T* quantized,
    float* error_output,
    size_t num_elements) {
    
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    
    const float orig_val = static_cast<float>(original[idx]);
    const float quant_val = static_cast<float>(quantized[idx]);
    
    // Compute absolute and relative errors
    const float abs_error = compute_absolute_error(orig_val, quant_val);
    const float rel_error = compute_relative_error(orig_val, quant_val);
    
    // Store both errors (pack into single float)
    error_output[idx * 2] = abs_error;
    error_output[idx * 2 + 1] = rel_error;
}

// Statistics computation kernel implementation
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
    if (histogram && bin_edges) {
        for (size_t i = 0; i < num_bins; ++i) {
            if (val >= bin_edges[i] && val < bin_edges[i + 1]) {
                atomicAdd(&histogram[i], 1);
                break;
            }
        }
    }
}

// Validation kernel implementation
template<typename T>
__global__ void validate_values_kernel(
    const T* input,
    uint32_t* invalid_count,
    size_t num_elements) {
    
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    
    const float val = static_cast<float>(input[idx]);
    
    // Check for invalid values (NaN or Inf)
    if (!is_valid_value(val)) {
        atomicAdd(invalid_count, 1);
    }
}

// Explicit template instantiations
template __global__ void dequantize_kernel<float>(
    const int8_t* input,
    float* output,
    const float* scales,
    const int8_t* zero_points,
    size_t num_elements,
    size_t elements_per_channel,
    bool per_channel,
    bool is_int4);

template __global__ void dequantize_kernel<half>(
    const int8_t* input,
    half* output,
    const float* scales,
    const int8_t* zero_points,
    size_t num_elements,
    size_t elements_per_channel,
    bool per_channel,
    bool is_int4);

template __global__ void compute_error_kernel<float>(
    const float* original,
    const float* quantized,
    float* error_output,
    size_t num_elements);

template __global__ void compute_error_kernel<half>(
    const half* original,
    const half* quantized,
    float* error_output,
    size_t num_elements);

template __global__ void compute_statistics_kernel<float>(
    const float* input,
    float* min_vals,
    float* max_vals,
    double* means,
    double* variances,
    uint32_t* histogram,
    const float* bin_edges,
    size_t num_elements,
    size_t num_bins);

template __global__ void compute_statistics_kernel<half>(
    const half* input,
    float* min_vals,
    float* max_vals,
    double* means,
    double* variances,
    uint32_t* histogram,
    const float* bin_edges,
    size_t num_elements,
    size_t num_bins);

template __global__ void validate_values_kernel<float>(
    const float* input,
    uint32_t* invalid_count,
    size_t num_elements);

template __global__ void validate_values_kernel<half>(
    const half* input,
    uint32_t* invalid_count,
    size_t num_elements);

} // namespace kernels
} // namespace deeppowers 