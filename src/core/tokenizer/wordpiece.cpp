#include "wordpiece.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <regex>
#include <queue>
#include <unicode/uchar.h>
#include <unicode/unistr.h>

namespace deeppowers {

WordPieceTokenizer::WordPieceTokenizer()
    : unk_token_("[UNK]")
    , prefix_("##")
    , max_input_chars_per_word_(200)
    , do_lower_case_(true)
    , do_basic_tokenize_(true)
    , tokenize_chinese_chars_(true)
    , strip_accents_(true) {
}

WordPieceTokenizer::~WordPieceTokenizer() = default;

void WordPieceTokenizer::train(const std::vector<std::string>& texts,
                              size_t vocab_size,
                              size_t min_frequency,
                              size_t max_subword_length) {
    // Clear existing vocabulary
    vocab_.clear();

    // Count word frequencies
    std::unordered_map<std::string, int32_t> word_freqs;
    for (const auto& text : texts) {
        auto words = basic_tokenize(text);
        for (const auto& word : words) {
            word_freqs[word]++;
        }
    }

    // Initialize with characters
    std::unordered_map<std::string, int32_t> subword_freqs;
    for (const auto& [word, freq] : word_freqs) {
        if (freq < min_frequency) continue;
        
        auto chars = split_to_unicode(word);
        for (const auto& c : chars) {
            subword_freqs[c] += freq;
        }
    }

    // Iteratively merge most frequent pairs
    while (vocab_.size() < vocab_size) {
        std::pair<std::string, std::string> best_pair;
        int32_t best_freq = 0;

        // Find best pair to merge
        for (const auto& [word, freq] : word_freqs) {
            if (freq < min_frequency) continue;
            
            auto subwords = wordpiece_tokenize(word);
            for (size_t i = 0; i < subwords.size() - 1; i++) {
                auto pair = std::make_pair(subwords[i], subwords[i + 1]);
                auto merged = subwords[i] + subwords[i + 1];
                
                if (merged.length() <= max_subword_length) {
                    int32_t pair_freq = freq;
                    if (pair_freq > best_freq) {
                        best_pair = pair;
                        best_freq = pair_freq;
                    }
                }
            }
        }

        if (best_freq < min_frequency) break;

        // Add merged token to vocabulary
        std::string merged_token = best_pair.first + best_pair.second;
        vocab_[merged_token] = vocab_.size();
    }
}

void WordPieceTokenizer::save_model(const std::string& path) const {
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for saving model: " + path);
    }

    // Save settings
    file << do_lower_case_ << "\n";
    file << do_basic_tokenize_ << "\n";
    file << tokenize_chinese_chars_ << "\n";
    file << strip_accents_ << "\n";
    file << max_input_chars_per_word_ << "\n";
    file << prefix_ << "\n";
    file << unk_token_ << "\n";

    // Save vocabulary
    file << vocab_.size() << "\n";
    for (const auto& [token, id] : vocab_) {
        file << token << "\t" << id << "\n";
    }
}

void WordPieceTokenizer::load_model(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for loading model: " + path);
    }

    // Load settings
    std::string line;
    std::getline(file, line); do_lower_case_ = (line == "1");
    std::getline(file, line); do_basic_tokenize_ = (line == "1");
    std::getline(file, line); tokenize_chinese_chars_ = (line == "1");
    std::getline(file, line); strip_accents_ = (line == "1");
    std::getline(file, line); max_input_chars_per_word_ = std::stoul(line);
    std::getline(file, prefix_);
    std::getline(file, unk_token_);

    // Load vocabulary
    vocab_.clear();
    size_t vocab_size;
    file >> vocab_size;
    std::string token;
    int32_t id;
    for (size_t i = 0; i < vocab_size; i++) {
        file >> token >> id;
        vocab_[token] = id;
    }
}

std::vector<std::string> WordPieceTokenizer::tokenize(const std::string& text) const {
    std::vector<std::string> output_tokens;
    
    // First do basic tokenization if enabled
    std::vector<std::string> tokens;
    if (do_basic_tokenize_) {
        tokens = basic_tokenize(text);
    } else {
        tokens = whitespace_tokenize(text);
    }

    // Then do WordPiece tokenization
    for (const auto& token : tokens) {
        if (token.length() > max_input_chars_per_word_) {
            output_tokens.push_back(unk_token_);
            continue;
        }

        auto sub_tokens = wordpiece_tokenize(token);
        output_tokens.insert(output_tokens.end(), sub_tokens.begin(), sub_tokens.end());
    }

    return output_tokens;
}

std::vector<std::string> WordPieceTokenizer::split_to_unicode(const std::string& text) const {
    std::vector<std::string> result;
    icu::UnicodeString ustr(text.c_str());
    
    for (int32_t i = 0; i < ustr.length(); i++) {
        char buf[8];
        int32_t len = 0;
        U8_APPEND_UNSAFE(buf, len, ustr.char32At(i));
        buf[len] = '\0';
        result.emplace_back(buf);
    }
    
    return result;
}

bool WordPieceTokenizer::is_chinese_char(char32_t cp) const {
    return (cp >= 0x4E00 && cp <= 0x9FFF) ||
           (cp >= 0x3400 && cp <= 0x4DBF) ||
           (cp >= 0x20000 && cp <= 0x2A6DF) ||
           (cp >= 0x2A700 && cp <= 0x2B73F) ||
           (cp >= 0x2B740 && cp <= 0x2B81F) ||
           (cp >= 0x2B820 && cp <= 0x2CEAF) ||
           (cp >= 0xF900 && cp <= 0xFAFF) ||
           (cp >= 0x2F800 && cp <= 0x2FA1F);
}

std::vector<std::string> WordPieceTokenizer::whitespace_tokenize(const std::string& text) const {
    std::vector<std::string> tokens;
    std::istringstream iss(text);
    std::string token;
    while (iss >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

std::vector<std::string> WordPieceTokenizer::basic_tokenize(const std::string& text) const {
    std::vector<std::string> tokens;
    
    // First split into unicode characters
    auto chars = split_to_unicode(text);
    
    // Process each character
    std::string current_token;
    for (const auto& c : chars) {
        icu::UnicodeString ustr(c.c_str());
        char32_t cp = ustr.char32At(0);
        
        // Handle Chinese characters
        if (tokenize_chinese_chars_ && is_chinese_char(cp)) {
            if (!current_token.empty()) {
                tokens.push_back(current_token);
                current_token.clear();
            }
            tokens.push_back(c);
            continue;
        }
        
        // Handle whitespace and punctuation
        if (u_isspace(cp) || u_ispunct(cp)) {
            if (!current_token.empty()) {
                tokens.push_back(current_token);
                current_token.clear();
            }
            tokens.push_back(c);
            continue;
        }
        
        // Accumulate characters for normal tokens
        current_token += c;
    }
    
    // Add final token if any
    if (!current_token.empty()) {
        tokens.push_back(current_token);
    }
    
    return tokens;
}

std::vector<std::string> WordPieceTokenizer::wordpiece_tokenize(const std::string& token) const {
    std::vector<std::string> sub_tokens;
    
    // Handle empty token
    if (token.empty()) return sub_tokens;
    
    // Convert to lowercase if needed
    std::string word = token;
    if (do_lower_case_) {
        std::transform(word.begin(), word.end(), word.begin(), ::tolower);
    }
    
    // Try to find the longest matching subword
    size_t start = 0;
    while (start < word.length()) {
        size_t end = word.length();
        bool found = false;
        
        while (start < end) {
            std::string substr = word.substr(start, end - start);
            if (start > 0) {
                substr = prefix_ + substr;
            }
            
            if (vocab_.find(substr) != vocab_.end()) {
                sub_tokens.push_back(substr);
                start = end;
                found = true;
                break;
            }
            end--;
        }
        
        if (!found) {
            sub_tokens.push_back(unk_token_);
            break;
        }
    }
    
    return sub_tokens;
}

} // namespace deeppowers 