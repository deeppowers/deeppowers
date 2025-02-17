#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include "bpe.hpp"
#include "wordpiece.hpp"
#include "tokenizer_kernels.cuh"
#include "../utils/thread_pool.hpp"
#include "../utils/memory_pool.hpp"
#include "../utils/string_pool.hpp"

namespace deeppowers {

// Forward declarations
class VocabManager;
struct GPUVocabTable;

// Tokenizer type enumeration
enum class TokenizerType {
    BPE,
    WordPiece
};

// Device type enumeration
enum class DeviceType {
    CPU,
    GPU
};

class Tokenizer {
public:
    Tokenizer(TokenizerType type = TokenizerType::BPE);
    ~Tokenizer();

    // Initialize tokenizer with vocabulary file
    void initialize(const std::string& vocab_path);

    // Train tokenizer on input texts
    void train(const std::vector<std::string>& texts, 
               size_t vocab_size,
               size_t min_frequency = 2);

    // Save/load tokenizer state
    void save(const std::string& path) const;
    void load(const std::string& path);

    // Encode text to token ids with memory optimization
    std::vector<int32_t> encode(std::string_view text) const;

    // Decode token ids back to text with string pooling
    std::string_view decode(const std::vector<int32_t>& tokens) const;

    // Batch processing methods with memory optimization
    std::vector<std::vector<int32_t>> encode_batch(
        const std::vector<std::string_view>& texts,
        bool pad_to_max_length = false) const;
    
    std::vector<std::string_view> decode_batch(
        const std::vector<std::vector<int32_t>>& token_batches) const;

    // Parallel batch processing methods
    std::vector<std::vector<int32_t>> encode_batch_parallel(
        const std::vector<std::string_view>& texts,
        bool pad_to_max_length = false,
        size_t num_threads = 0) const;
    
    std::vector<std::string_view> decode_batch_parallel(
        const std::vector<std::vector<int32_t>>& token_batches,
        size_t num_threads = 0) const;

    // GPU-accelerated batch processing methods
    std::vector<std::vector<int32_t>> encode_batch_gpu(
        const std::vector<std::string_view>& texts,
        bool pad_to_max_length = false) const;
    
    // Get vocabulary size
    size_t vocab_size() const;

    // Special token getters
    int32_t pad_token_id() const { return pad_token_id_; }
    int32_t eos_token_id() const { return eos_token_id_; }
    int32_t bos_token_id() const { return bos_token_id_; }
    int32_t unk_token_id() const { return unk_token_id_; }

    // Configuration
    void set_tokenizer_type(TokenizerType type);
    TokenizerType get_tokenizer_type() const { return tokenizer_type_; }
    
    // Device configuration
    void set_device_type(DeviceType device_type);
    DeviceType get_device_type() const { return device_type_; }
    bool is_gpu_available() const;

private:
    // Internal tokenization helpers
    std::vector<std::string_view> pre_tokenize(std::string_view text) const;
    std::vector<int32_t> encode_tokens(const std::vector<std::string_view>& tokens) const;
    
    // Parallel processing helpers
    void process_batch_range(
        const std::vector<std::string_view>& texts,
        size_t start,
        size_t end,
        std::vector<std::vector<int32_t>>& results) const;
        
    void process_decode_range(
        const std::vector<std::vector<int32_t>>& token_batches,
        size_t start,
        size_t end,
        std::vector<std::string_view>& results) const;
    
    // GPU processing helpers
    void initialize_gpu_vocab() const;
    void free_gpu_vocab() const;
    
    // Memory management
    void* allocate_memory(size_t size) const;
    void deallocate_memory(void* ptr) const;
    
    // String pooling
    std::string_view intern_string(const std::string& str) const;
    std::string_view intern_string(std::string_view str) const;
    
    // Vocabulary management
    std::unique_ptr<VocabManager> vocab_manager_;
    std::unique_ptr<BPETokenizer> bpe_tokenizer_;
    std::unique_ptr<WordPieceTokenizer> wordpiece_tokenizer_;
    
    // Special token IDs
    int32_t pad_token_id_;
    int32_t eos_token_id_;
    int32_t bos_token_id_;
    int32_t unk_token_id_;
    
    // Tokenization settings
    bool add_bos_token_;
    bool add_eos_token_;
    size_t max_token_length_;
    TokenizerType tokenizer_type_;
    DeviceType device_type_;
    
    // Memory pool for temporary allocations
    mutable MemoryPool* memory_pool_;
    
    // GPU resources
    mutable std::unique_ptr<GPUVocabTable> gpu_vocab_;
    mutable bool gpu_initialized_;
};

} // namespace deeppowers 