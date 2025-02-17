#include "graph_kernels.cuh"

namespace deeppowers {
namespace kernels {

// Matrix multiplication kernel implementation
template<typename T>
__global__ void matmul_kernel(
    const T* A,
    const T* B,
    T* C,
    int M,
    int N,
    int K,
    T alpha,
    T beta) {
    
    // Calculate the position of the output element handled by the current thread
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        T sum = 0;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

// Layer normalization kernel implementation
template<typename T>
__global__ void layernorm_kernel(
    const T* input,
    const T* gamma,
    const T* beta,
    T* output,
    int batch_size,
    int hidden_size,
    float eps) {
    
    // Each thread block processes a sequence
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    // Use shared memory to calculate mean and variance
    extern __shared__ float s_data[];
    float* s_mean = s_data;
    float* s_variance = s_data + blockDim.x;
    
    // Calculate mean
    float local_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        local_sum += static_cast<float>(input[batch_idx * hidden_size + i]);
    }
    s_mean[tid] = local_sum;
    __syncthreads();
    
    // Reduce sum
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_mean[tid] += s_mean[tid + stride];
        }
        __syncthreads();
    }
    
    // Calculate mean
    if (tid == 0) {
        s_mean[0] /= hidden_size;
    }
    __syncthreads();
    
    // Calculate variance
    local_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float diff = static_cast<float>(input[batch_idx * hidden_size + i]) - s_mean[0];
        local_sum += diff * diff;
    }
    s_variance[tid] = local_sum;
    __syncthreads();
    
    // Reduce sum
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_variance[tid] += s_variance[tid + stride];
        }
        __syncthreads();
    }
    
    // Calculate variance
    if (tid == 0) {
        s_variance[0] = sqrtf(s_variance[0] / hidden_size + eps);
    }
    __syncthreads();
    
    // Normalize and apply scaling and offset
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        const int idx = batch_idx * hidden_size + i;
        float normalized = (static_cast<float>(input[idx]) - s_mean[0]) / s_variance[0];
        output[idx] = static_cast<T>(normalized * static_cast<float>(gamma[i]) + 
                                   static_cast<float>(beta[i]));
    }
}

// Attention calculation kernel implementation
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
    int head_dim) {
    
    // Each thread block processes a attention head
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (query_idx >= seq_length) return;
    
    // Calculate attention scores
    extern __shared__ float s_data[];
    float* s_scores = s_data;
    float* s_max = s_data + seq_length;
    float* s_sum = s_max + 1;
    
    // Calculate the dot product of a query vector with all key vectors
    float max_score = -INFINITY;
    for (int key_idx = 0; key_idx < seq_length; ++key_idx) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            const int q_idx = ((batch_idx * num_heads + head_idx) * seq_length + query_idx) * head_dim + d;
            const int k_idx = ((batch_idx * num_heads + head_idx) * seq_length + key_idx) * head_dim + d;
            score += static_cast<float>(query[q_idx]) * static_cast<float>(key[k_idx]);
        }
        
        // Apply mask
        if (mask) {
            const int mask_idx = batch_idx * seq_length * seq_length + query_idx * seq_length + key_idx;
            score += static_cast<float>(mask[mask_idx]) * -10000.0f;
        }
        
        s_scores[key_idx] = score;
        max_score = fmaxf(max_score, score);
    }
    
    // Calculate softmax
    float sum = 0.0f;
    for (int i = 0; i < seq_length; ++i) {
        s_scores[i] = __expf(s_scores[i] - max_score);
        sum += s_scores[i];
    }
    
    // Calculate weighted sum
    for (int d = 0; d < head_dim; ++d) {
        float weighted_sum = 0.0f;
        for (int i = 0; i < seq_length; ++i) {
            const int v_idx = ((batch_idx * num_heads + head_idx) * seq_length + i) * head_dim + d;
            weighted_sum += s_scores[i] * static_cast<float>(value[v_idx]) / sum;
        }
        const int out_idx = ((batch_idx * num_heads + head_idx) * seq_length + query_idx) * head_dim + d;
        output[out_idx] = static_cast<T>(weighted_sum);
    }
}

// Activation function kernel implementation
template<typename T>
__global__ void activation_kernel(
    const T* input,
    T* output,
    int size,
    const char* activation_type) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    const float x = static_cast<float>(input[idx]);
    float result;
    
    // Calculate based on activation function type
    if (activation_type[0] == 'r') {  // ReLU
        result = fmaxf(x, 0.0f);
    } else if (activation_type[0] == 'g') {  // GELU
        result = gelu(x);
    } else if (activation_type[0] == 't') {  // Tanh
        result = tanhf(x);
    } else {  // Default to linear
        result = x;
    }
    
    output[idx] = static_cast<T>(result);
}

// Element-wise operation kernel implementation
template<typename T>
__global__ void elementwise_kernel(
    const T* input_a,
    const T* input_b,
    T* output,
    int size,
    const char* op_type) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    const float a = static_cast<float>(input_a[idx]);
    const float b = static_cast<float>(input_b[idx]);
    float result;
    
    // Calculate based on operation type
    switch (op_type[0]) {
        case 'a':  // add
            result = a + b;
            break;
        case 'm':  // multiply
            result = a * b;
            break;
        case 'd':  // divide
            result = a / b;
            break;
        case 's':  // subtract
            result = a - b;
            break;
        default:
            result = a;
    }
    
    output[idx] = static_cast<T>(result);
}

// Tensor transpose kernel implementation
template<typename T>
__global__ void transpose_kernel(
    const T* input,
    T* output,
    const int* dims,
    const int* perm,
    int rank) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate total number of elements
    int total_size = 1;
    for (int i = 0; i < rank; ++i) {
        total_size *= dims[i];
    }
    
    if (idx >= total_size) return;
    
    // Calculate original index
    int old_idx = idx;
    int old_indices[8];  // Assume maximum dimension is 8
    for (int i = rank - 1; i >= 0; --i) {
        old_indices[i] = old_idx % dims[i];
        old_idx /= dims[i];
    }
    
    // Calculate transposed index
    int new_idx = 0;
    int stride = 1;
    for (int i = rank - 1; i >= 0; --i) {
        new_idx += old_indices[perm[i]] * stride;
        stride *= dims[perm[i]];
    }
    
    output[new_idx] = input[idx];
}

// Explicit template instantiation
template __global__ void matmul_kernel<float>(const float*, const float*, float*, int, int, int, float, float);
template __global__ void matmul_kernel<half>(const half*, const half*, half*, int, int, int, half, half);

template __global__ void layernorm_kernel<float>(const float*, const float*, const float*, float*, int, int, float);
template __global__ void layernorm_kernel<half>(const half*, const half*, const half*, half*, int, int, float);

template __global__ void attention_kernel<float>(const float*, const float*, const float*, const float*, float*, int, int, int, int);
template __global__ void attention_kernel<half>(const half*, const half*, const half*, const half*, half*, int, int, int, int);

template __global__ void activation_kernel<float>(const float*, float*, int, const char*);
template __global__ void activation_kernel<half>(const half*, half*, int, const char*);

template __global__ void elementwise_kernel<float>(const float*, const float*, float*, int, const char*);
template __global__ void elementwise_kernel<half>(const half*, const half*, half*, int, const char*);

template __global__ void transpose_kernel<float>(const float*, float*, const int*, const int*, int);
template __global__ void transpose_kernel<half>(const half*, half*, const int*, const int*, int);

} // namespace kernels
} // namespace deeppowers 