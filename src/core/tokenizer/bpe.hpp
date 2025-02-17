#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <queue>
#include <utility>

namespace deeppowers {

// BPE merge rule representation
struct MergeRule {
    std::string first;
    std::string second;
    std::string result;
    int32_t priority;

    bool operator<(const MergeRule& other) const {
        return priority > other.priority;  // Higher priority comes first
    }
};

class BPETokenizer {
public:
    BPETokenizer();
    ~BPETokenizer();

    // Train BPE on input text
    void train(const std::vector<std::string>& texts, size_t vocab_size, size_t min_frequency = 2);

    // Save/load merge rules
    void save_rules(const std::string& path) const;
    void load_rules(const std::string& path);

    // Tokenize text using trained rules
    std::vector<std::string> tokenize(const std::string& text) const;

private:
    // Internal helper methods
    std::vector<std::string> get_initial_tokens(const std::string& text) const;
    void count_token_pairs(const std::vector<std::string>& tokens);
    MergeRule find_best_merge() const;
    void apply_merge(const MergeRule& rule);
    
    // Data structures for BPE
    std::unordered_map<std::string, int32_t> token_frequencies_;
    std::vector<MergeRule> merge_rules_;
    std::unordered_map<std::pair<std::string, std::string>, int32_t, PairHash> pair_frequencies_;
    
    // Settings
    size_t max_token_length_;
    bool enable_caching_;
    
    // Cache for optimization
    mutable std::unordered_map<std::string, std::vector<std::string>> tokenization_cache_;
};

// Hash function for string pairs
struct PairHash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

} // namespace deeppowers 