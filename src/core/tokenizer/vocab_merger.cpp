#include "vocab_merger.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <queue>

namespace deeppowers {

VocabMerger::VocabMerger() = default;
VocabMerger::~VocabMerger() = default;

void VocabMerger::add_vocabulary(const std::string& vocab_path, float weight) {
    VocabEntry entry;
    entry.weight = weight;
    
    // Load vocabulary from file
    std::ifstream file(vocab_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open vocabulary file: " + vocab_path);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        int32_t freq;
        
        if (iss >> token >> freq) {
            entry.tokens[token] = freq;
        }
    }
    
    vocabularies_.push_back(std::move(entry));
}

void VocabMerger::add_vocabulary(
    const std::unordered_map<std::string, int32_t>& vocab,
    float weight) {
    
    VocabEntry entry;
    entry.tokens = vocab;
    entry.weight = weight;
    vocabularies_.push_back(std::move(entry));
}

void VocabMerger::merge(
    const std::string& output_path,
    size_t max_vocab_size,
    MergeStrategy strategy) {
    
    // Merge frequencies according to strategy
    std::unordered_map<std::string, float> merged_freqs;
    switch (strategy) {
        case MergeStrategy::UNION:
            merge_union(merged_freqs);
            break;
        case MergeStrategy::INTERSECTION:
            merge_intersection(merged_freqs);
            break;
        case MergeStrategy::WEIGHTED_UNION:
            merge_weighted(merged_freqs);
            break;
        case MergeStrategy::FREQUENCY_BASED:
            merge_frequency(merged_freqs);
            break;
    }
    
    // Sort tokens by frequency
    using TokenFreq = std::pair<std::string, float>;
    std::vector<TokenFreq> sorted_tokens(merged_freqs.begin(), merged_freqs.end());
    std::sort(sorted_tokens.begin(), sorted_tokens.end(),
              [](const TokenFreq& a, const TokenFreq& b) {
                  return a.second > b.second;
              });
    
    // Truncate to max_vocab_size
    if (sorted_tokens.size() > max_vocab_size) {
        sorted_tokens.resize(max_vocab_size);
    }
    
    // Save merged vocabulary
    std::ofstream file(output_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open output file: " + output_path);
    }
    
    for (const auto& [token, freq] : sorted_tokens) {
        file << token << "\t" << static_cast<int32_t>(freq) << "\n";
    }
}

void VocabMerger::merge_union(std::unordered_map<std::string, float>& merged_freqs) const {
    for (const auto& vocab : vocabularies_) {
        for (const auto& [token, freq] : vocab.tokens) {
            merged_freqs[token] = std::max(merged_freqs[token],
                                         static_cast<float>(freq) * vocab.weight);
        }
    }
}

void VocabMerger::merge_intersection(std::unordered_map<std::string, float>& merged_freqs) const {
    if (vocabularies_.empty()) return;
    
    // Start with first vocabulary
    for (const auto& [token, freq] : vocabularies_[0].tokens) {
        merged_freqs[token] = static_cast<float>(freq) * vocabularies_[0].weight;
    }
    
    // Intersect with other vocabularies
    for (size_t i = 1; i < vocabularies_.size(); ++i) {
        const auto& vocab = vocabularies_[i];
        auto it = merged_freqs.begin();
        while (it != merged_freqs.end()) {
            auto vocab_it = vocab.tokens.find(it->first);
            if (vocab_it == vocab.tokens.end()) {
                it = merged_freqs.erase(it);
            } else {
                it->second = std::min(it->second,
                                    static_cast<float>(vocab_it->second) * vocab.weight);
                ++it;
            }
        }
    }
}

void VocabMerger::merge_weighted(std::unordered_map<std::string, float>& merged_freqs) const {
    for (const auto& vocab : vocabularies_) {
        for (const auto& [token, freq] : vocab.tokens) {
            merged_freqs[token] += static_cast<float>(freq) * vocab.weight;
        }
    }
}

void VocabMerger::merge_frequency(std::unordered_map<std::string, float>& merged_freqs) const {
    // Create priority queue for each vocabulary
    using TokenFreq = std::pair<std::string, float>;
    std::vector<std::priority_queue<TokenFreq>> queues;
    
    for (const auto& vocab : vocabularies_) {
        std::priority_queue<TokenFreq> queue;
        for (const auto& [token, freq] : vocab.tokens) {
            queue.push({token, static_cast<float>(freq) * vocab.weight});
        }
        queues.push_back(std::move(queue));
    }
    
    // Merge using priority queues
    while (!queues.empty()) {
        float max_freq = 0.0f;
        std::string max_token;
        size_t max_queue_idx = 0;
        
        // Find highest frequency token
        for (size_t i = 0; i < queues.size(); ++i) {
            if (!queues[i].empty()) {
                const auto& top = queues[i].top();
                if (top.second > max_freq) {
                    max_freq = top.second;
                    max_token = top.first;
                    max_queue_idx = i;
                }
            }
        }
        
        if (max_freq == 0.0f) break;
        
        // Add to merged frequencies
        merged_freqs[max_token] = max_freq;
        
        // Remove from queue
        queues[max_queue_idx].pop();
        if (queues[max_queue_idx].empty()) {
            queues.erase(queues.begin() + max_queue_idx);
        }
    }
}

} // namespace deeppowers 