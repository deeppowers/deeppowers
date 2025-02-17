#pragma once

#include <string>
#include <vector>
#include <unordered_map>

namespace deeppowers {

class VocabManager {
public:
    VocabManager();
    ~VocabManager();

    // Load vocabulary from file
    void load_vocab(const std::string& vocab_path);

    // Token to ID conversion
    int32_t token_to_id(const std::string& token) const;
    std::string id_to_token(int32_t id) const;

    // Vocabulary information
    size_t vocab_size() const { return vocab_.size(); }
    bool contains_token(const std::string& token) const;

    // Special tokens management
    void add_special_token(const std::string& token, int32_t id);
    bool is_special_token(int32_t id) const;
    bool is_special_token(const std::string& token) const;

private:
    // Token-ID mappings
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::vector<std::string> id_to_token_;
    
    // Special tokens set
    std::unordered_map<std::string, int32_t> special_tokens_;
    
    // Vocabulary
    std::vector<std::string> vocab_;
};

} // namespace deeppowers 