#include "vocab_manager.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace deeppowers {

VocabManager::VocabManager() = default;
VocabManager::~VocabManager() = default;

void VocabManager::load_vocab(const std::string& vocab_path) {
    std::ifstream file(vocab_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open vocabulary file: " + vocab_path);
    }

    // Clear existing vocabulary
    vocab_.clear();
    token_to_id_.clear();
    id_to_token_.clear();

    // Read vocabulary file
    std::string line;
    int32_t id = 0;
    while (std::getline(file, line)) {
        // Remove whitespace
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        if (!line.empty()) {
            // Add token to vocabulary
            vocab_.push_back(line);
            token_to_id_[line] = id;
            id_to_token_.push_back(line);
            id++;
        }
    }
}

int32_t VocabManager::token_to_id(const std::string& token) const {
    auto it = token_to_id_.find(token);
    if (it != token_to_id_.end()) {
        return it->second;
    }

    // Check special tokens
    auto special_it = special_tokens_.find(token);
    if (special_it != special_tokens_.end()) {
        return special_it->second;
    }

    throw std::runtime_error("Token not found in vocabulary: " + token);
}

std::string VocabManager::id_to_token(int32_t id) const {
    if (id < 0 || id >= static_cast<int32_t>(id_to_token_.size())) {
        // Check special tokens
        for (const auto& pair : special_tokens_) {
            if (pair.second == id) {
                return pair.first;
            }
        }
        throw std::runtime_error("Token ID out of range: " + std::to_string(id));
    }
    return id_to_token_[id];
}

bool VocabManager::contains_token(const std::string& token) const {
    return token_to_id_.find(token) != token_to_id_.end() ||
           special_tokens_.find(token) != special_tokens_.end();
}

void VocabManager::add_special_token(const std::string& token, int32_t id) {
    special_tokens_[token] = id;
}

bool VocabManager::is_special_token(int32_t id) const {
    for (const auto& pair : special_tokens_) {
        if (pair.second == id) {
            return true;
        }
    }
    return false;
}

bool VocabManager::is_special_token(const std::string& token) const {
    return special_tokens_.find(token) != special_tokens_.end();
}

} // namespace deeppowers 