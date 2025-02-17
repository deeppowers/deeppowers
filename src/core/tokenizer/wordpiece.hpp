#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace deeppowers {

class WordPieceTokenizer {
public:
    WordPieceTokenizer();
    ~WordPieceTokenizer();

    // Train WordPiece model on input texts
    void train(const std::vector<std::string>& texts,
               size_t vocab_size,
               size_t min_frequency = 2,
               size_t max_subword_length = 20);

    // Save/load model
    void save_model(const std::string& path) const;
    void load_model(const std::string& path);

    // Tokenize text using trained model
    std::vector<std::string> tokenize(const std::string& text) const;

private:
    // Internal helper methods
    std::vector<std::string> split_to_unicode(const std::string& text) const;
    bool is_chinese_char(char32_t cp) const;
    std::vector<std::string> whitespace_tokenize(const std::string& text) const;
    std::vector<std::string> basic_tokenize(const std::string& text) const;
    std::vector<std::string> wordpiece_tokenize(const std::string& token) const;

    // Vocabulary and model parameters
    std::unordered_map<std::string, int32_t> vocab_;
    std::string unk_token_;
    std::string prefix_;
    size_t max_input_chars_per_word_;
    
    // Settings
    bool do_lower_case_;
    bool do_basic_tokenize_;
    bool tokenize_chinese_chars_;
    bool strip_accents_;
};

} // namespace deeppowers 