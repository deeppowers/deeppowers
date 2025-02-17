#pragma once

#include <vector>
#include <string>
#include <cstdint>

namespace deeppowers {

// Forward declarations
struct GPUVocabTable;

// Initialize GPU vocabulary table
cudaError_t initializeGPUVocab(
    const std::vector<std::string>& tokens,
    const std::vector<int>& ids,
    GPUVocabTable& gpu_vocab);

// Free GPU vocabulary table
cudaError_t freeGPUVocab(GPUVocabTable& gpu_vocab);

// GPU-accelerated tokenization
std::vector<std::vector<int32_t>> tokenizeOnGPU(
    const std::string& text,
    const GPUVocabTable& gpu_vocab);

} // namespace deeppowers 