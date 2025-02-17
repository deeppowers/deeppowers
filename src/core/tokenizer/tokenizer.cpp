#include "tokenizer.hpp"
#include "vocab_manager.hpp"
#include "../utils/thread_pool.hpp"
#include <stdexcept>
#include <algorithm>
#include <regex>
#include <thread>
#include <future>
#include <fstream>

namespace deeppowers {

Tokenizer::Tokenizer(TokenizerType type)
    : vocab_manager_(std::make_unique<VocabManager>())
    , bpe_tokenizer_(std::make_unique<BPETokenizer>())
    , wordpiece_tokenizer_(std::make_unique<WordPieceTokenizer>())
    , pad_token_id_(0)
    , eos_token_id_(1)
    , bos_token_id_(2)
    , unk_token_id_(3)
    , add_bos_token_(true)
    , add_eos_token_(true)
    , max_token_length_(256)
    , tokenizer_type_(type)
    , memory_pool_(new MemoryPool(4096)) {  // 4KB blocks
    device_type_ = DeviceType::CPU;
    gpu_initialized_ = false;
    gpu_vocab_ = nullptr;
}

Tokenizer::~Tokenizer() {
    delete memory_pool_;
}

void Tokenizer::set_tokenizer_type(TokenizerType type) {
    if (type != tokenizer_type_) {
        tokenizer_type_ = type;
        if (type == TokenizerType::BPE) {
            bpe_tokenizer_ = std::make_unique<BPETokenizer>();
            wordpiece_tokenizer_.reset();
        } else {
            wordpiece_tokenizer_ = std::make_unique<WordPieceTokenizer>();
            bpe_tokenizer_.reset();
        }
    }
}

void Tokenizer::initialize(const std::string& vocab_path) {
    vocab_manager_->load_vocabulary(vocab_path);
    
    if (tokenizer_type_ == TokenizerType::BPE) {
        bpe_tokenizer_->initialize(vocab_manager_.get());
    } else {
        wordpiece_tokenizer_->initialize(vocab_manager_.get());
    }
}

void Tokenizer::train(const std::vector<std::string>& texts, 
                     size_t vocab_size,
                     size_t min_frequency) {
    // Pre-process texts and collect statistics
    std::vector<std::vector<std::string_view>> pre_tokenized;
    pre_tokenized.reserve(texts.size());
    
    for (const auto& text : texts) {
        pre_tokenized.push_back(pre_tokenize(text));
    }
    
    if (tokenizer_type_ == TokenizerType::BPE) {
        bpe_tokenizer_->train(pre_tokenized, vocab_size, min_frequency);
    } else {
        wordpiece_tokenizer_->train(pre_tokenized, vocab_size, min_frequency);
    }
}

void Tokenizer::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open file for saving: " + path);
    }
    
    // Save tokenizer type
    out.write(reinterpret_cast<const char*>(&tokenizer_type_), sizeof(TokenizerType));
    
    // Save vocabulary
    vocab_manager_->save(out);
    
    // Save tokenizer-specific data
    if (tokenizer_type_ == TokenizerType::BPE) {
        bpe_tokenizer_->save(out);
    } else {
        wordpiece_tokenizer_->save(out);
    }
}

void Tokenizer::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open file for loading: " + path);
    }
    
    // Load tokenizer type
    TokenizerType loaded_type;
    in.read(reinterpret_cast<char*>(&loaded_type), sizeof(TokenizerType));
    set_tokenizer_type(loaded_type);
    
    // Load vocabulary
    vocab_manager_->load(in);
    
    // Load tokenizer-specific data
    if (tokenizer_type_ == TokenizerType::BPE) {
        bpe_tokenizer_->load(in);
    } else {
        wordpiece_tokenizer_->load(in);
    }
}

std::vector<int32_t> Tokenizer::encode(std::string_view text) const {
    // Pre-tokenize the text
    auto tokens = pre_tokenize(text);
    
    // Encode tokens to IDs
    auto token_ids = encode_tokens(tokens);
    
    // Add special tokens if needed
    if (add_bos_token_) {
        token_ids.insert(token_ids.begin(), bos_token_id_);
    }
    if (add_eos_token_) {
        token_ids.push_back(eos_token_id_);
    }
    
    return token_ids;
}

std::string_view Tokenizer::decode(const std::vector<int32_t>& tokens) const {
    std::string result;
    result.reserve(tokens.size() * 4); // Estimate average token length
    
    for (size_t i = 0; i < tokens.size(); ++i) {
        // Skip special tokens
        if (tokens[i] == bos_token_id_ || tokens[i] == eos_token_id_ || 
            tokens[i] == pad_token_id_) {
            continue;
        }
        
        // Get token text
        auto token_text = vocab_manager_->id_to_token(tokens[i]);
        if (i > 0 && !token_text.empty() && token_text[0] != 'Ġ') {
            result += ' ';
        }
        if (!token_text.empty() && token_text[0] == 'Ġ') {
            result += token_text.substr(1);
        } else {
            result += token_text;
        }
    }
    
    return intern_string(result);
}

std::vector<std::vector<int32_t>> Tokenizer::encode_batch(
    const std::vector<std::string_view>& texts,
    bool pad_to_max_length) const {
    
    std::vector<std::vector<int32_t>> results;
    results.reserve(texts.size());
    
    // Process each text
    size_t max_length = 0;
    for (const auto& text : texts) {
        auto encoded = encode(text);
        max_length = std::max(max_length, encoded.size());
        results.push_back(std::move(encoded));
    }
    
    // Pad if requested
    if (pad_to_max_length) {
        for (auto& tokens : results) {
            tokens.resize(max_length, pad_token_id_);
        }
    }
    
    return results;
}

std::vector<std::string_view> Tokenizer::decode_batch(
    const std::vector<std::vector<int32_t>>& token_batches) const {
    
    std::vector<std::string_view> texts;
    texts.reserve(token_batches.size());
    
    for (const auto& tokens : token_batches) {
        texts.push_back(decode(tokens));
    }
    
    return texts;
}

size_t Tokenizer::vocab_size() const {
    return vocab_manager_->vocab_size();
}

std::vector<std::string_view> Tokenizer::pre_tokenize(std::string_view text) const {
    std::vector<std::string_view> tokens;
    
    // Basic whitespace tokenization
    size_t start = 0;
    size_t end = 0;
    
    while (end < text.size()) {
        // Skip whitespace
        while (start < text.size() && std::isspace(text[start])) {
            ++start;
        }
        
        // Find end of token
        end = start;
        while (end < text.size() && !std::isspace(text[end])) {
            ++end;
        }
        
        // Add token if found
        if (start < end) {
            tokens.push_back(text.substr(start, end - start));
        }
        
        start = end;
    }
    
    return tokens;
}

std::vector<int32_t> Tokenizer::encode_tokens(
    const std::vector<std::string_view>& tokens) const {
    
    std::vector<int32_t> token_ids;
    token_ids.reserve(tokens.size());
    
    for (const auto& token : tokens) {
        if (tokenizer_type_ == TokenizerType::BPE) {
            auto subtoken_ids = bpe_tokenizer_->encode_token(token);
            token_ids.insert(token_ids.end(), subtoken_ids.begin(), subtoken_ids.end());
        } else {
            auto subtoken_ids = wordpiece_tokenizer_->encode_token(token);
            token_ids.insert(token_ids.end(), subtoken_ids.begin(), subtoken_ids.end());
        }
    }
    
    return token_ids;
}

std::vector<std::vector<int32_t>> Tokenizer::encode_batch_parallel(
    const std::vector<std::string_view>& texts,
    bool pad_to_max_length,
    size_t num_threads) const {
    
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
    }
    
    // Create thread pool
    ThreadPool pool(num_threads);
    
    // Prepare results vector
    std::vector<std::vector<int32_t>> results(texts.size());
    
    // Calculate batch size per thread
    size_t batch_size = (texts.size() + num_threads - 1) / num_threads;
    
    // Submit tasks to thread pool
    std::vector<std::future<void>> futures;
    for (size_t i = 0; i < texts.size(); i += batch_size) {
        size_t end = std::min(i + batch_size, texts.size());
        futures.push_back(pool.enqueue([this, &texts, &results, i, end]() {
            process_batch_range(texts, i, end, results);
        }));
    }
    
    // Wait for all tasks to complete
    for (auto& future : futures) {
        future.get();
    }
    
    // Pad if requested
    if (pad_to_max_length) {
        size_t max_length = 0;
        for (const auto& tokens : results) {
            max_length = std::max(max_length, tokens.size());
        }
        for (auto& tokens : results) {
            tokens.resize(max_length, pad_token_id_);
        }
    }
    
    return results;
}

std::vector<std::string> Tokenizer::decode_batch_parallel(
    const std::vector<std::vector<int32_t>>& token_batches,
    size_t num_threads) const {
    
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
    }
    
    // Create thread pool
    ThreadPool pool(num_threads);
    
    // Prepare result vector
    std::vector<std::string> results(token_batches.size());
    
    // Calculate batch size for each thread
    size_t batch_size = (token_batches.size() + num_threads - 1) / num_threads;
    std::vector<std::future<void>> futures;
    
    // Submit tasks to thread pool
    for (size_t i = 0; i < num_threads; ++i) {
        size_t start = i * batch_size;
        size_t end = std::min(start + batch_size, token_batches.size());
        
        if (start >= token_batches.size()) break;
        
        futures.push_back(pool.enqueue(
            [this, &token_batches, start, end, &results] {
                process_decode_range(token_batches, start, end, results);
            }
        ));
    }
    
    // Wait for all tasks to complete
    for (auto& future : futures) {
        future.get();
    }
    
    return results;
}

void Tokenizer::process_batch_range(
    const std::vector<std::string_view>& texts,
    size_t start,
    size_t end,
    std::vector<std::vector<int32_t>>& results) const {
    
    for (size_t i = start; i < end; ++i) {
        results[i] = encode(texts[i]);
    }
}

void Tokenizer::process_decode_range(
    const std::vector<std::vector<int32_t>>& token_batches,
    size_t start,
    size_t end,
    std::vector<std::string>& results) const {
    
    for (size_t i = start; i < end; ++i) {
        results[i] = decode(token_batches[i]);
    }
}

void* Tokenizer::allocate_memory(size_t size) const {
    return memory_pool_->allocate(size);
}

void Tokenizer::deallocate_memory(void* ptr) const {
    memory_pool_->deallocate(ptr);
}

std::string_view Tokenizer::intern_string(const std::string& str) const {
    return vocab_manager_->intern_string(str);
}

std::string_view Tokenizer::intern_string(std::string_view str) const {
    return vocab_manager_->intern_string(str);
}

void Tokenizer::set_device_type(DeviceType device_type) {
    if (device_type == DeviceType::GPU && !is_gpu_available()) {
        throw std::runtime_error("GPU is not available");
    }
    
    if (device_type_ != device_type) {
        // Clean up old device resources
        if (device_type_ == DeviceType::GPU) {
            free_gpu_vocab();
        }
        
        device_type_ = device_type;
        
        // Initialize new device resources
        if (device_type_ == DeviceType::GPU) {
            initialize_gpu_vocab();
        }
    }
}

bool Tokenizer::is_gpu_available() const {
    int device_count;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    return (error == cudaSuccess && device_count > 0);
}

void Tokenizer::initialize_gpu_vocab() const {
    if (gpu_initialized_) return;
    
    // Get vocabulary data
    auto tokens = vocab_manager_->get_all_tokens();
    std::vector<int> ids(tokens.size());
    for (size_t i = 0; i < tokens.size(); i++) {
        ids[i] = vocab_manager_->token_to_id(tokens[i]);
    }
    
    // Create and initialize GPU vocabulary
    gpu_vocab_ = std::make_unique<GPUVocabTable>();
    cudaError_t error = initializeGPUVocab(tokens, ids, *gpu_vocab_);
    
    if (error != cudaSuccess) {
        gpu_vocab_.reset();
        throw std::runtime_error("Failed to initialize GPU vocabulary: " + 
                               std::string(cudaGetErrorString(error)));
    }
    
    gpu_initialized_ = true;
}

void Tokenizer::free_gpu_vocab() const {
    if (!gpu_initialized_) return;
    
    if (gpu_vocab_) {
        freeGPUVocab(*gpu_vocab_);
        gpu_vocab_.reset();
    }
    
    gpu_initialized_ = false;
}

std::vector<std::vector<int32_t>> Tokenizer::encode_batch_gpu(
    const std::vector<std::string_view>& texts,
    bool pad_to_max_length) const {
    
    if (device_type_ != DeviceType::GPU) {
        throw std::runtime_error("GPU acceleration is not enabled");
    }
    
    if (!gpu_initialized_) {
        initialize_gpu_vocab();
    }
    
    // Concatenate all texts with space separator
    std::string combined_text;
    size_t total_length = 0;
    for (const auto& text : texts) {
        total_length += text.length() + 1;  // +1 for space
    }
    combined_text.reserve(total_length);
    
    for (size_t i = 0; i < texts.size(); i++) {
        if (i > 0) combined_text += ' ';
        combined_text.append(texts[i].data(), texts[i].length());
    }
    
    // Perform GPU tokenization
    auto token_ids = tokenizeOnGPU(combined_text, *gpu_vocab_);
    
    // Pad if requested
    if (pad_to_max_length) {
        size_t max_length = 0;
        for (const auto& tokens : token_ids) {
            max_length = std::max(max_length, tokens.size());
        }
        
        for (auto& tokens : token_ids) {
            tokens.resize(max_length, pad_token_id_);
        }
    }
    
    return token_ids;
}

} // namespace deeppowers 