#include "bpe.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <queue>
#include <regex>

namespace deeppowers {

BPETokenizer::BPETokenizer()
    : max_token_length_(100)
    , enable_caching_(true) {
}

BPETokenizer::~BPETokenizer() = default;

void BPETokenizer::train(const std::vector<std::string>& texts, size_t vocab_size, size_t min_frequency) {
    // Clear existing data
    token_frequencies_.clear();
    merge_rules_.clear();
    pair_frequencies_.clear();
    
    // Count initial tokens
    for (const auto& text : texts) {
        auto tokens = get_initial_tokens(text);
        for (const auto& token : tokens) {
            token_frequencies_[token]++;
        }
    }
    
    // Remove low frequency tokens
    for (auto it = token_frequencies_.begin(); it != token_frequencies_.end();) {
        if (it->second < min_frequency) {
            it = token_frequencies_.erase(it);
        } else {
            ++it;
        }
    }
    
    // Main BPE training loop
    while (token_frequencies_.size() < vocab_size) {
        // Count pair frequencies
        count_token_pairs(texts);
        
        // Find best pair to merge
        auto best_merge = find_best_merge();
        if (best_merge.priority <= 0) break;
        
        // Apply merge
        apply_merge(best_merge);
        merge_rules_.push_back(best_merge);
    }
}

void BPETokenizer::save_rules(const std::string& path) const {
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for saving merge rules: " + path);
    }
    
    // Save merge rules
    for (const auto& rule : merge_rules_) {
        file << rule.first << "\t" << rule.second << "\t" 
             << rule.result << "\t" << rule.priority << "\n";
    }
}

void BPETokenizer::load_rules(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for loading merge rules: " + path);
    }
    
    merge_rules_.clear();
    
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        MergeRule rule;
        iss >> rule.first >> rule.second >> rule.result >> rule.priority;
        merge_rules_.push_back(rule);
    }
}

std::vector<std::string> BPETokenizer::tokenize(const std::string& text) const {
    // Check cache first
    if (enable_caching_) {
        auto it = tokenization_cache_.find(text);
        if (it != tokenization_cache_.end()) {
            return it->second;
        }
    }
    
    // Get initial tokens
    std::vector<std::string> tokens = get_initial_tokens(text);
    
    // Apply merge rules
    for (const auto& rule : merge_rules_) {
        for (size_t i = 0; i < tokens.size() - 1; ++i) {
            if (tokens[i] == rule.first && tokens[i + 1] == rule.second) {
                tokens[i] = rule.result;
                tokens.erase(tokens.begin() + i + 1);
                --i;
            }
        }
    }
    
    // Cache result
    if (enable_caching_) {
        tokenization_cache_[text] = tokens;
    }
    
    return tokens;
}

std::vector<std::string> BPETokenizer::get_initial_tokens(const std::string& text) const {
    std::vector<std::string> tokens;
    
    // Split into words first
    std::regex word_regex(R"(\w+|\s+|[^\w\s])");
    std::sregex_iterator it(text.begin(), text.end(), word_regex);
    std::sregex_iterator end;
    
    while (it != end) {
        std::string word = it->str();
        // Split words into characters
        for (char c : word) {
            tokens.push_back(std::string(1, c));
        }
        ++it;
    }
    
    return tokens;
}

void BPETokenizer::count_token_pairs(const std::vector<std::string>& tokens) {
    pair_frequencies_.clear();
    
    for (size_t i = 0; i < tokens.size() - 1; ++i) {
        auto pair = std::make_pair(tokens[i], tokens[i + 1]);
        pair_frequencies_[pair]++;
    }
}

MergeRule BPETokenizer::find_best_merge() const {
    MergeRule best_rule;
    best_rule.priority = 0;
    
    for (const auto& pair : pair_frequencies_) {
        if (pair.second > best_rule.priority) {
            best_rule.first = pair.first.first;
            best_rule.second = pair.first.second;
            best_rule.result = pair.first.first + pair.first.second;
            best_rule.priority = pair.second;
        }
    }
    
    return best_rule;
}

void BPETokenizer::apply_merge(const MergeRule& rule) {
    // Update token frequencies
    int32_t new_freq = 0;
    for (const auto& pair : pair_frequencies_) {
        if (pair.first.first == rule.first && pair.first.second == rule.second) {
            new_freq += pair.second;
        }
    }
    
    token_frequencies_[rule.result] = new_freq;
    token_frequencies_[rule.first] -= new_freq;
    token_frequencies_[rule.second] -= new_freq;
    
    // Remove tokens if frequency becomes 0
    if (token_frequencies_[rule.first] == 0) {
        token_frequencies_.erase(rule.first);
    }
    if (token_frequencies_[rule.second] == 0) {
        token_frequencies_.erase(rule.second);
    }
}

} // namespace deeppowers 