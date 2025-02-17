#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace deeppowers {

class VocabMerger {
public:
    VocabMerger();
    ~VocabMerger();

    // Add vocabulary from file
    void add_vocabulary(const std::string& vocab_path, float weight = 1.0f);

    // Add vocabulary from memory
    void add_vocabulary(const std::unordered_map<std::string, int32_t>& vocab,
                       float weight = 1.0f);

    // Merge vocabularies with different strategies
    enum class MergeStrategy {
        UNION,              // Take union of all vocabs
        INTERSECTION,       // Take intersection of all vocabs
        WEIGHTED_UNION,     // Take union with weighted frequencies
        FREQUENCY_BASED     // Keep most frequent tokens
    };

    // Merge and save result
    void merge(const std::string& output_path,
              size_t max_vocab_size,
              MergeStrategy strategy = MergeStrategy::WEIGHTED_UNION);

private:
    // Internal structures
    struct VocabEntry {
        std::unordered_map<std::string, int32_t> tokens;
        float weight;
    };

    // Helper methods
    void merge_union(std::unordered_map<std::string, float>& merged_freqs) const;
    void merge_intersection(std::unordered_map<std::string, float>& merged_freqs) const;
    void merge_weighted(std::unordered_map<std::string, float>& merged_freqs) const;
    void merge_frequency(std::unordered_map<std::string, float>& merged_freqs) const;

    // Storage for vocabularies
    std::vector<VocabEntry> vocabularies_;
};

} // namespace deeppowers 