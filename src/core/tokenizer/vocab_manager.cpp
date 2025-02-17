#include "vocab_manager.hpp"
#include <fstream>
#include <algorithm>
#include <stdexcept>

namespace deeppowers {

// Constructor
VocabManager::VocabManager()
    : next_id_(0)
    , string_pool_(std::make_unique<StringPool>()) {
    // Add special tokens
    add_special_token("<pad>", 0);  // Padding token
    add_special_token("<eos>", 1);  // End of sequence
    add_special_token("<bos>", 2);  // Beginning of sequence
    add_special_token("<unk>", 3);  // Unknown token
    next_id_ = 4;  // Start normal tokens after special tokens
}

// Add a new token to vocabulary
void VocabManager::add_token(const std::string& token) {
    if (!contains_token(token)) {
        auto token_view = intern_string(token);
        token_to_id_[token_view] = next_id_;
        id_to_token_[next_id_] = token_view;
        next_id_++;
    }
}

// Add a special token with specific ID
void VocabManager::add_special_token(const std::string& token, int32_t id) {
    if (id < 0 || contains_token(token)) {
        throw std::invalid_argument("Invalid special token or ID");
    }
    
    auto token_view = intern_string(token);
    token_to_id_[token_view] = id;
    id_to_token_[id] = token_view;
    special_token_ids_.insert(id);
    
    next_id_ = std::max(next_id_, id + 1);
}

// Check if token exists in vocabulary
bool VocabManager::contains_token(const std::string& token) const {
    return token_to_id_.find(token) != token_to_id_.end();
}

// Check if token exists in vocabulary
bool VocabManager::contains_token(std::string_view token) const {
    return token_to_id_.find(token) != token_to_id_.end();
}

// Get token ID for a given token
int32_t VocabManager::token_to_id(const std::string& token) const {
    auto it = token_to_id_.find(token);
    return it != token_to_id_.end() ? it->second : unk_token_id();
}

// Get token ID for a given token
int32_t VocabManager::token_to_id(std::string_view token) const {
    auto it = token_to_id_.find(token);
    return it != token_to_id_.end() ? it->second : unk_token_id();
}

// Get token for a given ID
std::string_view VocabManager::id_to_token(int32_t id) const {
    auto it = id_to_token_.find(id);
    if (it != id_to_token_.end()) {
        return it->second;
    }
    return id_to_token_.at(unk_token_id());
}

// Check if an ID represents a special token
bool VocabManager::is_special_token(int32_t id) const {
    return special_token_ids_.find(id) != special_token_ids_.end();
}

// Get all tokens in vocabulary
std::vector<std::string> VocabManager::get_all_tokens() const {
    std::vector<std::string> tokens;
    tokens.reserve(token_to_id_.size());
    
    for (const auto& [token, _] : token_to_id_) {
        tokens.push_back(std::string(token));
    }
    
    return tokens;
}

// Get vocabulary size
size_t VocabManager::vocab_size() const {
    return token_to_id_.size();
}

// Clear vocabulary
void VocabManager::clear() {
    token_to_id_.clear();
    id_to_token_.clear();
    special_token_ids_.clear();
    string_pool_ = std::make_unique<StringPool>();
    next_id_ = 0;
}

// Save vocabulary to file
void VocabManager::save(std::ostream& out) const {
    // Save number of tokens
    size_t num_tokens = token_to_id_.size();
    out.write(reinterpret_cast<const char*>(&num_tokens), sizeof(size_t));
    
    // Save tokens and their IDs
    for (const auto& [token, id] : token_to_id_) {
        // Save token
        size_t token_len = token.length();
        out.write(reinterpret_cast<const char*>(&token_len), sizeof(size_t));
        out.write(token.data(), token_len);
        
        // Save ID
        out.write(reinterpret_cast<const char*>(&id), sizeof(int32_t));
        
        // Save whether it's a special token
        bool is_special = is_special_token(id);
        out.write(reinterpret_cast<const char*>(&is_special), sizeof(bool));
    }
}

// Load vocabulary from file
void VocabManager::load(std::istream& in) {
    clear();
    
    // Load number of tokens
    size_t num_tokens;
    in.read(reinterpret_cast<char*>(&num_tokens), sizeof(size_t));
    
    // Load tokens and their IDs
    for (size_t i = 0; i < num_tokens; ++i) {
        // Load token
        size_t token_len;
        in.read(reinterpret_cast<char*>(&token_len), sizeof(size_t));
        std::string token(token_len, '\0');
        in.read(&token[0], token_len);
        
        // Load ID
        int32_t id;
        in.read(reinterpret_cast<char*>(&id), sizeof(int32_t));
        
        // Load whether it's a special token
        bool is_special;
        in.read(reinterpret_cast<char*>(&is_special), sizeof(bool));
        
        // Add to vocabulary
        if (is_special) {
            add_special_token(token, id);
        } else {
            auto token_view = intern_string(token);
            token_to_id_[token_view] = id;
            id_to_token_[id] = token_view;
        }
        
        next_id_ = std::max(next_id_, id + 1);
    }
}

// Load vocabulary from text file
void VocabManager::load_vocabulary(const std::string& vocab_path) {
    std::ifstream file(vocab_path);
    if (!file) {
        throw std::runtime_error("Failed to open vocabulary file: " + vocab_path);
    }
    
    clear();
    
    std::string line;
    while (std::getline(file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;
        
        // Add token to vocabulary
        add_token(line);
    }
}

// String pooling
std::string_view VocabManager::intern_string(const std::string& str) const {
    return string_pool_->intern(str);
}

std::string_view VocabManager::intern_string(std::string_view str) const {
    return string_pool_->intern(str);
}

} // namespace deeppowers 