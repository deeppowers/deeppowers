#include "tokenizer.hpp"
#include "vocab_manager.hpp"
#include "bpe.hpp"
#include "wordpiece.hpp"
#include <stdexcept>
#include <algorithm>
#include <regex>
#include <thread>
#include <future>

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
    , max_token_length_(96)
    , tokenizer_type_(type)
    , memory_pool_(new MemoryPool(4096)) {  // 4KB blocks
}

Tokenizer::~Tokenizer() {
    delete memory_pool_;
}

void Tokenizer::set_tokenizer_type(TokenizerType type) {
    tokenizer_type_ = type;
}

void Tokenizer::initialize(const std::string& vocab_path) {
    // Load vocabulary
    vocab_manager_->load_vocab(vocab_path);
    
    // Add special tokens
    vocab_manager_->add_special_token("<pad>", pad_token_id_);
    vocab_manager_->add_special_token("<eos>", eos_token_id_);
    vocab_manager_->add_special_token("<bos>", bos_token_id_);
    vocab_manager_->add_special_token("<unk>", unk_token_id_);
}

void Tokenizer::train(const std::vector<std::string>& texts, 
                     size_t vocab_size,
                     size_t min_frequency) {
    switch (tokenizer_type_) {
        case TokenizerType::BPE:
            bpe_tokenizer_->train(texts, vocab_size, min_frequency);
            break;
        case TokenizerType::WordPiece:
            wordpiece_tokenizer_->train(texts, vocab_size, min_frequency);
            break;
    }
}

void Tokenizer::save(const std::string& path) const {
    switch (tokenizer_type_) {
        case TokenizerType::BPE:
            bpe_tokenizer_->save_rules(path + ".bpe");
            break;
        case TokenizerType::WordPiece:
            wordpiece_tokenizer_->save_model(path + ".wordpiece");
            break;
    }
}

void Tokenizer::load(const std::string& path) {
    switch (tokenizer_type_) {
        case TokenizerType::BPE:
            bpe_tokenizer_->load_rules(path + ".bpe");
            break;
        case TokenizerType::WordPiece:
            wordpiece_tokenizer_->load_model(path + ".wordpiece");
            break;
    }
}

std::vector<int32_t> Tokenizer::encode(std::string_view text) const {
    // Pre-tokenize text into subwords
    std::vector<std::string_view> tokens;
    switch (tokenizer_type_) {
        case TokenizerType::BPE:
            tokens = bpe_tokenizer_->tokenize(text);
            break;
        case TokenizerType::WordPiece:
            tokens = wordpiece_tokenizer_->tokenize(text);
            break;
    }
    
    // Convert tokens to IDs using memory pool
    std::vector<int32_t>* token_ids = new(allocate_memory(sizeof(std::vector<int32_t>)))
        std::vector<int32_t>();
    token_ids->reserve(tokens.size() + 2);  // Reserve space for BOS/EOS
    
    // Add BOS token if needed
    if (add_bos_token_) {
        token_ids->push_back(bos_token_id_);
    }
    
    // Add token IDs
    for (const auto& token : tokens) {
        if (vocab_manager_->contains_token(std::string(token))) {
            token_ids->push_back(vocab_manager_->token_to_id(std::string(token)));
        } else {
            token_ids->push_back(unk_token_id_);
        }
    }
    
    // Add EOS token if needed
    if (add_eos_token_) {
        token_ids->push_back(eos_token_id_);
    }
    
    // Truncate if needed
    if (token_ids->size() > max_token_length_) {
        token_ids->resize(max_token_length_);
        // Make sure we still have EOS token if needed
        if (add_eos_token_) {
            token_ids->back() = eos_token_id_;
        }
    }
    
    // Move result to output
    std::vector<int32_t> result = std::move(*token_ids);
    
    // Clean up
    token_ids->~vector();
    deallocate_memory(token_ids);
    
    return result;
}

std::string_view Tokenizer::decode(const std::vector<int32_t>& tokens) const {
    std::string result;
    bool is_first = true;
    
    for (int32_t token_id : tokens) {
        // Skip special tokens
        if (vocab_manager_->is_special_token(token_id)) {
            continue;
        }
        
        std::string_view token = intern_string(vocab_manager_->id_to_token(token_id));
        
        // Add space between normal tokens
        if (!is_first && !token.empty() && token[0] != '#') {
            result += ' ';
        }
        
        result += token;
        is_first = false;
    }
    
    return intern_string(result);
}

std::vector<std::vector<int32_t>> Tokenizer::encode_batch(
    const std::vector<std::string_view>& texts,
    bool pad_to_max_length) const {
    
    std::vector<std::vector<int32_t>> batch_tokens;
    batch_tokens.reserve(texts.size());
    
    // Encode all texts
    size_t max_length = 0;
    for (const auto& text : texts) {
        auto tokens = encode(text);
        max_length = std::max(max_length, tokens.size());
        batch_tokens.push_back(std::move(tokens));
    }
    
    // Pad sequences if requested
    if (pad_to_max_length) {
        for (auto& tokens : batch_tokens) {
            if (tokens.size() < max_length) {
                tokens.resize(max_length, pad_token_id_);
            }
        }
    }
    
    return batch_tokens;
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

std::vector<std::string> Tokenizer::pre_tokenize(const std::string& text) const {
    std::vector<std::string> tokens;
    
    // Split into words first
    std::regex word_regex(R"(\w+|\s+|[^\w\s])");
    std::sregex_iterator it(text.begin(), text.end(), word_regex);
    std::sregex_iterator end;
    
    while (it != end) {
        std::string word = it->str();
        if (!word.empty()) {
            tokens.push_back(word);
        }
        ++it;
    }
    
    return tokens;
}

std::vector<int32_t> Tokenizer::encode_tokens(const std::vector<std::string>& tokens) const {
    std::vector<int32_t> token_ids;
    token_ids.reserve(tokens.size());
    
    for (const auto& token : tokens) {
        if (vocab_manager_->contains_token(token)) {
            token_ids.push_back(vocab_manager_->token_to_id(token));
        } else {
            token_ids.push_back(unk_token_id_);
        }
    }
    
    return token_ids;
}

std::vector<std::vector<int32_t>> Tokenizer::encode_batch_parallel(
    const std::vector<std::string>& texts,
    bool pad_to_max_length,
    size_t num_threads) const {
    
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
    }
    
    // Create thread pool
    ThreadPool pool(num_threads);
    
    // Prepare result vector
    std::vector<std::vector<int32_t>> results(texts.size());
    
    // Calculate batch size for each thread
    size_t batch_size = (texts.size() + num_threads - 1) / num_threads;
    std::vector<std::future<void>> futures;
    
    // Submit tasks to thread pool
    for (size_t i = 0; i < num_threads; ++i) {
        size_t start = i * batch_size;
        size_t end = std::min(start + batch_size, texts.size());
        
        if (start >= texts.size()) break;
        
        futures.push_back(pool.enqueue(
            [this, &texts, start, end, &results] {
                process_batch_range(texts, start, end, results);
            }
        ));
    }
    
    // Wait for all tasks to complete
    for (auto& future : futures) {
        future.get();
    }
    
    // Pad sequences if requested
    if (pad_to_max_length) {
        size_t max_length = 0;
        for (const auto& tokens : results) {
            max_length = std::max(max_length, tokens.size());
        }
        
        for (auto& tokens : results) {
            if (tokens.size() < max_length) {
                tokens.resize(max_length, pad_token_id_);
            }
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
    const std::vector<std::string>& texts,
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
    return get_string_pool().intern(str);
}

std::string_view Tokenizer::intern_string(std::string_view str) const {
    return get_string_pool().intern(str);
}

} // namespace deeppowers 