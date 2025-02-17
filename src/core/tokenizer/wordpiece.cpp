#include "wordpiece.hpp"
#include "vocab_manager.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <regex>
#include <queue>
#include <unicode/uchar.h>
#include <unicode/unistr.h>

namespace deeppowers {

WordPieceTokenizer::WordPieceTokenizer()
    : vocab_manager_(nullptr)
    , max_word_length_(100)
    , unknown_token_("##UNK##")
    , unk_token_("[UNK]")
    , prefix_("##")
    , max_input_chars_per_word_(200)
    , do_lower_case_(true)
    , do_basic_tokenize_(true)
    , tokenize_chinese_chars_(true)
    , strip_accents_(true) {
}

WordPieceTokenizer::~WordPieceTokenizer() = default;

void WordPieceTokenizer::initialize(VocabManager* vocab_manager) {
    vocab_manager_ = vocab_manager;
    build_trie();
}

void WordPieceTokenizer::train(
    const std::vector<std::vector<std::string_view>>& pre_tokenized,
    size_t vocab_size,
    size_t min_frequency) {
    
    // Count subword frequencies
    std::unordered_map<std::string, size_t> subword_freqs;
    
    // Initialize with whole words and their subwords
    for (const auto& tokens : pre_tokenized) {
        for (const auto& token : tokens) {
            std::string word(token);
            if (word.length() > max_word_length_) continue;
            
            // Add whole word
            subword_freqs[word]++;
            
            // Add all possible subwords
            for (size_t start = 0; start < word.length(); ++start) {
                for (size_t len = 1; len <= word.length() - start; ++len) {
                    std::string subword = word.substr(start, len);
                    if (start > 0) {
                        subword = "##" + subword;
                    }
                    subword_freqs[subword]++;
                }
            }
        }
    }
    
    // Sort subwords by frequency
    std::vector<std::pair<std::string, size_t>> subword_freq_pairs;
    for (const auto& [subword, freq] : subword_freqs) {
        if (freq >= min_frequency) {
            subword_freq_pairs.push_back({subword, freq});
        }
    }
    
    std::sort(subword_freq_pairs.begin(), subword_freq_pairs.end(),
        [](const auto& a, const auto& b) {
            return a.second > b.second;
        }
    );
    
    // Add top subwords to vocabulary
    size_t num_tokens = std::min(vocab_size, subword_freq_pairs.size());
    for (size_t i = 0; i < num_tokens; ++i) {
        vocab_manager_->add_token(subword_freq_pairs[i].first);
    }
    
    // Build trie for efficient tokenization
    build_trie();
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
    file << vocab_manager_->size() << "\n";
    for (const auto& [token, id] : vocab_manager_->get_vocab()) {
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
    vocab_manager_->clear();
    size_t vocab_size;
    file >> vocab_size;
    std::string token;
    int32_t id;
    for (size_t i = 0; i < vocab_size; i++) {
        file >> token >> id;
        vocab_manager_->add_token(token);
    }

    // Rebuild trie
    build_trie();
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
            
            if (vocab_manager_->contains_token(substr)) {
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

void WordPieceTokenizer::build_trie() {
    trie_root_ = std::make_unique<TrieNode>();
    
    // Get all tokens from vocabulary
    auto tokens = vocab_manager_->get_all_tokens();
    
    // Add each token to trie
    for (const auto& token : tokens) {
        TrieNode* node = trie_root_.get();
        
        // Handle ##prefix for continuation subwords
        size_t start = 0;
        bool is_continuation = false;
        if (token.length() >= 2 && token.substr(0, 2) == "##") {
            start = 2;
            is_continuation = true;
        }
        
        // Add characters to trie
        for (size_t i = start; i < token.length(); ++i) {
            char c = token[i];
            if (node->children.find(c) == node->children.end()) {
                node->children[c] = std::make_unique<TrieNode>();
            }
            node = node->children[c].get();
        }
        
        // Mark end of token
        node->is_token = true;
        node->token = token;
        node->token_id = vocab_manager_->token_to_id(token);
        node->is_continuation = is_continuation;
    }
}

} // namespace deeppowers 