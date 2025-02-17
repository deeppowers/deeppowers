#include "bpe.hpp"
#include "vocab_manager.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <queue>
#include <regex>
#include <unordered_map>

namespace deeppowers {

BPETokenizer::BPETokenizer()
    : max_token_length_(100)
    , enable_caching_(true)
    , vocab_manager_(nullptr) {
}

BPETokenizer::~BPETokenizer() = default;

void BPETokenizer::initialize(VocabManager* vocab_manager) {
    vocab_manager_ = vocab_manager;
    load_merge_rules();
}

void BPETokenizer::train(
    const std::vector<std::vector<std::string_view>>& pre_tokenized,
    size_t vocab_size,
    size_t min_frequency) {
    
    // Count initial character frequencies
    std::unordered_map<std::string, size_t> char_freqs;
    for (const auto& tokens : pre_tokenized) {
        for (const auto& token : tokens) {
            for (char c : std::string(token)) {
                char_freqs[std::string(1, c)]++;
            }
        }
    }
    
    // Initialize vocabulary with characters
    for (const auto& [ch, freq] : char_freqs) {
        if (freq >= min_frequency) {
            vocab_manager_->add_token(ch);
        }
    }
    
    // Initialize token splits
    std::unordered_map<std::string, std::vector<std::string>> token_splits;
    for (const auto& tokens : pre_tokenized) {
        for (const auto& token : tokens) {
            std::string token_str(token);
            if (token_splits.find(token_str) == token_splits.end()) {
                // Split token into characters
                std::vector<std::string> chars;
                for (char c : token_str) {
                    chars.push_back(std::string(1, c));
                }
                token_splits[token_str] = chars;
            }
        }
    }
    
    // Count pair frequencies
    auto pair_freqs = count_pair_frequencies(token_splits);
    
    // Merge pairs until vocab_size is reached
    while (vocab_manager_->vocab_size() < vocab_size && !pair_freqs.empty()) {
        // Get most frequent pair
        auto best_pair = get_most_frequent_pair(pair_freqs);
        if (best_pair.second < min_frequency) break;
        
        // Create new token
        std::string new_token = best_pair.first.first + best_pair.first.second;
        vocab_manager_->add_token(new_token);
        
        // Add merge rule
        merge_rules_.push_back(best_pair.first);
        
        // Update token splits
        update_token_splits(token_splits, best_pair.first);
        
        // Update pair frequencies
        pair_freqs = count_pair_frequencies(token_splits);
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

std::unordered_map<std::pair<std::string, std::string>, size_t, PairHash>
BPETokenizer::count_pair_frequencies(
    const std::unordered_map<std::string, std::vector<std::string>>& token_splits) const {
    
    std::unordered_map<std::pair<std::string, std::string>, size_t, PairHash> pair_freqs;
    
    for (const auto& [token, splits] : token_splits) {
        for (size_t i = 0; i < splits.size() - 1; ++i) {
            pair_freqs[{splits[i], splits[i + 1]}]++;
        }
    }
    
    return pair_freqs;
}

std::pair<std::pair<std::string, std::string>, size_t>
BPETokenizer::get_most_frequent_pair(
    const std::unordered_map<std::pair<std::string, std::string>, size_t, PairHash>& pair_freqs) const {
    
    auto max_it = std::max_element(
        pair_freqs.begin(),
        pair_freqs.end(),
        [](const auto& p1, const auto& p2) {
            return p1.second < p2.second;
        }
    );
    
    return *max_it;
}

void BPETokenizer::update_token_splits(
    std::unordered_map<std::string, std::vector<std::string>>& token_splits,
    const std::pair<std::string, std::string>& pair) const {
    
    for (auto& [token, splits] : token_splits) {
        size_t i = 0;
        while (i < splits.size() - 1) {
            if (splits[i] == pair.first && splits[i + 1] == pair.second) {
                // Merge parts
                splits[i] = splits[i] + splits[i + 1];
                splits.erase(splits.begin() + i + 1);
            } else {
                ++i;
            }
        }
    }
}

void BPETokenizer::load_merge_rules() {
    merge_rules_.clear();
    
    // Get all tokens from vocabulary
    auto tokens = vocab_manager_->get_all_tokens();
    
    // Sort tokens by length (longer tokens first)
    std::sort(tokens.begin(), tokens.end(),
        [](const std::string& a, const std::string& b) {
            return a.length() > b.length();
        }
    );
    
    // Create merge rules
    for (const auto& token : tokens) {
        if (token.length() > 1) {
            // Find the best split point
            size_t best_split = 1;
            for (size_t i = 1; i < token.length(); ++i) {
                std::string first = token.substr(0, i);
                std::string second = token.substr(i);
                if (vocab_manager_->contains_token(first) &&
                    vocab_manager_->contains_token(second)) {
                    best_split = i;
                    break;
                }
            }
            
            // Add merge rule
            merge_rules_.push_back({
                token.substr(0, best_split),
                token.substr(best_split)
            });
        }
    }
}

} // namespace deeppowers 