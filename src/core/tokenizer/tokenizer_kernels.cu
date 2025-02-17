#include "tokenizer_kernels.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace deeppowers {

// Constants for GPU processing
constexpr int BLOCK_SIZE = 256;
constexpr int MAX_SEQUENCE_LENGTH = 2048;

// Device-side vocabulary lookup table
struct GPUVocabTable {
    int* token_ids;        // Token ID lookup table
    char* token_data;      // Token string data
    int* token_offsets;    // Token string offsets
    int num_tokens;        // Number of tokens in vocabulary
};

// CUDA kernel for parallel pre-tokenization
__global__ void preTokenizeKernel(
    const char* input_text,
    int text_length,
    int* token_starts,
    int* token_lengths,
    int* num_tokens) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= text_length) return;
    
    // Shared memory for token counting
    __shared__ int shared_count;
    if (threadIdx.x == 0) {
        shared_count = 0;
    }
    __syncthreads();
    
    // Check if current character is a token boundary
    bool is_boundary = false;
    if (tid == 0) {
        is_boundary = true;
    } else {
        char curr = input_text[tid];
        char prev = input_text[tid - 1];
        is_boundary = (isspace(prev) && !isspace(curr));
    }
    
    // Mark token boundaries
    if (is_boundary) {
        int token_idx = atomicAdd(&shared_count, 1);
        if (token_idx < MAX_SEQUENCE_LENGTH) {
            token_starts[token_idx] = tid;
            
            // Calculate token length
            int length = 0;
            int pos = tid;
            while (pos < text_length && !isspace(input_text[pos])) {
                length++;
                pos++;
            }
            token_lengths[token_idx] = length;
        }
    }
    __syncthreads();
    
    // Store total number of tokens
    if (threadIdx.x == 0) {
        *num_tokens = min(shared_count, MAX_SEQUENCE_LENGTH);
    }
}

// CUDA kernel for parallel BPE encoding
__global__ void bpeEncodeKernel(
    const char* input_text,
    const int* token_starts,
    const int* token_lengths,
    int num_tokens,
    const GPUVocabTable vocab,
    int* output_ids,
    int* output_lengths) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_tokens) return;
    
    // Get current token
    int start = token_starts[tid];
    int length = token_lengths[tid];
    
    // Shared memory for temporary token storage
    __shared__ char shared_token[MAX_SEQUENCE_LENGTH];
    
    // Copy token to shared memory
    for (int i = threadIdx.x; i < length; i += blockDim.x) {
        if (start + i < MAX_SEQUENCE_LENGTH) {
            shared_token[i] = input_text[start + i];
        }
    }
    __syncthreads();
    
    // Find longest matching subword
    int out_idx = 0;
    int pos = 0;
    
    while (pos < length && out_idx < MAX_SEQUENCE_LENGTH) {
        int best_length = 0;
        int best_id = vocab.token_ids[0];  // Default to UNK token
        
        // Try all possible subwords starting at current position
        for (int i = 0; i < vocab.num_tokens; i++) {
            int token_start = vocab.token_offsets[i];
            int token_length = vocab.token_offsets[i + 1] - token_start;
            
            if (token_length > length - pos) continue;
            
            // Check if subword matches
            bool matches = true;
            for (int j = 0; j < token_length; j++) {
                if (shared_token[pos + j] != vocab.token_data[token_start + j]) {
                    matches = false;
                    break;
                }
            }
            
            if (matches && token_length > best_length) {
                best_length = token_length;
                best_id = vocab.token_ids[i];
            }
        }
        
        // Add best matching subword
        output_ids[tid * MAX_SEQUENCE_LENGTH + out_idx] = best_id;
        out_idx++;
        pos += (best_length > 0) ? best_length : 1;
    }
    
    output_lengths[tid] = out_idx;
}

// Host-side functions for GPU acceleration
cudaError_t initializeGPUVocab(
    const std::vector<std::string>& tokens,
    const std::vector<int>& ids,
    GPUVocabTable& gpu_vocab) {
    
    // Allocate device memory for vocabulary
    cudaMalloc(&gpu_vocab.token_ids, ids.size() * sizeof(int));
    cudaMemcpy(gpu_vocab.token_ids, ids.data(), ids.size() * sizeof(int), cudaMemcpyHostToDevice);
    
    // Calculate total string data size and offsets
    std::vector<int> offsets(tokens.size() + 1, 0);
    int total_size = 0;
    for (size_t i = 0; i < tokens.size(); i++) {
        offsets[i] = total_size;
        total_size += tokens[i].length();
    }
    offsets[tokens.size()] = total_size;
    
    // Allocate and copy string data
    std::vector<char> token_data(total_size);
    for (size_t i = 0; i < tokens.size(); i++) {
        std::copy(tokens[i].begin(), tokens[i].end(), 
                 token_data.begin() + offsets[i]);
    }
    
    cudaMalloc(&gpu_vocab.token_data, total_size * sizeof(char));
    cudaMemcpy(gpu_vocab.token_data, token_data.data(), 
               total_size * sizeof(char), cudaMemcpyHostToDevice);
    
    cudaMalloc(&gpu_vocab.token_offsets, offsets.size() * sizeof(int));
    cudaMemcpy(gpu_vocab.token_offsets, offsets.data(), 
               offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
    
    gpu_vocab.num_tokens = tokens.size();
    
    return cudaGetLastError();
}

cudaError_t freeGPUVocab(GPUVocabTable& gpu_vocab) {
    cudaFree(gpu_vocab.token_ids);
    cudaFree(gpu_vocab.token_data);
    cudaFree(gpu_vocab.token_offsets);
    return cudaGetLastError();
}

// Host-side function for GPU-accelerated tokenization
std::vector<std::vector<int32_t>> tokenizeOnGPU(
    const std::string& text,
    const GPUVocabTable& gpu_vocab) {
    
    // Allocate device memory for input text
    char* d_text;
    cudaMalloc(&d_text, text.length() * sizeof(char));
    cudaMemcpy(d_text, text.data(), text.length() * sizeof(char), 
               cudaMemcpyHostToDevice);
    
    // Allocate device memory for pre-tokenization results
    int* d_token_starts;
    int* d_token_lengths;
    int* d_num_tokens;
    cudaMalloc(&d_token_starts, MAX_SEQUENCE_LENGTH * sizeof(int));
    cudaMalloc(&d_token_lengths, MAX_SEQUENCE_LENGTH * sizeof(int));
    cudaMalloc(&d_num_tokens, sizeof(int));
    
    // Launch pre-tokenization kernel
    int num_blocks = (text.length() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    preTokenizeKernel<<<num_blocks, BLOCK_SIZE>>>(
        d_text, text.length(), d_token_starts, d_token_lengths, d_num_tokens);
    
    // Get number of tokens
    int num_tokens;
    cudaMemcpy(&num_tokens, d_num_tokens, sizeof(int), cudaMemcpyDeviceToHost);
    
    // Allocate device memory for encoding results
    int* d_output_ids;
    int* d_output_lengths;
    cudaMalloc(&d_output_ids, num_tokens * MAX_SEQUENCE_LENGTH * sizeof(int));
    cudaMalloc(&d_output_lengths, num_tokens * sizeof(int));
    
    // Launch encoding kernel
    num_blocks = (num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE;
    bpeEncodeKernel<<<num_blocks, BLOCK_SIZE>>>(
        d_text, d_token_starts, d_token_lengths, num_tokens,
        gpu_vocab, d_output_ids, d_output_lengths);
    
    // Copy results back to host
    std::vector<int> output_lengths(num_tokens);
    cudaMemcpy(output_lengths.data(), d_output_lengths, 
               num_tokens * sizeof(int), cudaMemcpyDeviceToHost);
    
    std::vector<int> output_ids(num_tokens * MAX_SEQUENCE_LENGTH);
    cudaMemcpy(output_ids.data(), d_output_ids,
               num_tokens * MAX_SEQUENCE_LENGTH * sizeof(int), 
               cudaMemcpyDeviceToHost);
    
    // Convert to final format
    std::vector<std::vector<int32_t>> result(num_tokens);
    for (int i = 0; i < num_tokens; i++) {
        result[i].assign(
            output_ids.begin() + i * MAX_SEQUENCE_LENGTH,
            output_ids.begin() + i * MAX_SEQUENCE_LENGTH + output_lengths[i]
        );
    }
    
    // Free device memory
    cudaFree(d_text);
    cudaFree(d_token_starts);
    cudaFree(d_token_lengths);
    cudaFree(d_num_tokens);
    cudaFree(d_output_ids);
    cudaFree(d_output_lengths);
    
    return result;
}

} // namespace deeppowers 