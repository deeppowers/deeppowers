#include "batch.hpp"
#include <algorithm>
#include <numeric>

namespace deeppowers {

Batch::Batch(const std::vector<RequestPtr>& requests)
    : requests_(requests)
    , status_(BatchStatus::PENDING)
    , created_time_(std::chrono::system_clock::now())
    , start_time_(std::chrono::system_clock::time_point::min())
    , end_time_(std::chrono::system_clock::time_point::min()) {
    
    update_stats();
}

void Batch::set_status(BatchStatus status) {
    status_ = status;
    
    // Update timestamps based on status
    auto now = std::chrono::system_clock::now();
    switch (status) {
        case BatchStatus::PROCESSING:
            start_time_ = now;
            break;
        case BatchStatus::COMPLETED:
        case BatchStatus::FAILED:
            end_time_ = now;
            break;
        default:
            break;
    }
}

void Batch::mark_started() {
    set_status(BatchStatus::PROCESSING);
    for (auto& request : requests_) {
        request->mark_started();
    }
}

void Batch::mark_completed() {
    set_status(BatchStatus::COMPLETED);
    for (auto& request : requests_) {
        request->mark_completed();
    }
}

void Batch::mark_failed(const std::string& error_message) {
    set_status(BatchStatus::FAILED);
    for (auto& request : requests_) {
        request->mark_failed(error_message);
    }
}

void Batch::update_stats() {
    if (requests_.empty()) {
        stats_ = BatchStats{};
        return;
    }

    // Calculate sequence lengths
    std::vector<size_t> sequence_lengths;
    sequence_lengths.reserve(requests_.size());
    
    for (const auto& request : requests_) {
        // Here we assume there is a function to get the number of tokens in the sequence
        // The actual implementation needs to calculate the number of tokens using the tokenizer
        size_t length = request->prompt().length(); // Temporary use character length instead
        sequence_lengths.push_back(length);
    }

    // Calculate statistics
    stats_.max_sequence_length = *std::max_element(sequence_lengths.begin(), 
                                                 sequence_lengths.end());
    stats_.min_sequence_length = *std::min_element(sequence_lengths.begin(), 
                                                 sequence_lengths.end());
    
    size_t total_length = std::accumulate(sequence_lengths.begin(), 
                                        sequence_lengths.end(), 
                                        static_cast<size_t>(0));
    stats_.avg_sequence_length = static_cast<float>(total_length) / sequence_lengths.size();
    
    // Calculate padding tokens
    stats_.total_tokens = stats_.max_sequence_length * requests_.size();
    stats_.padding_tokens = stats_.total_tokens - total_length;
    stats_.padding_ratio = static_cast<float>(stats_.padding_tokens) / stats_.total_tokens;
}

bool Batch::is_compatible(const RequestPtr& request) const {
    if (!request) {
        return false;
    }

    // Check if the batch has already started processing
    if (status_ != BatchStatus::PENDING) {
        return false;
    }

    // Check if the sequence length is appropriate
    size_t sequence_length = request->prompt().length(); // Temporary use character length instead
    if (sequence_length > stats_.max_sequence_length) {
        // If the sequence length of the new request is longer, we need to evaluate the padding cost
        size_t new_padding = (sequence_length - stats_.max_sequence_length) * requests_.size();
        float new_padding_ratio = static_cast<float>(stats_.padding_tokens + new_padding) / 
                                (sequence_length * (requests_.size() + 1));
        
        // If the padding ratio is too high, we consider it incompatible
        if (new_padding_ratio > 0.3f) { // Configurable threshold
            return false;
        }
    }

    // Check other compatibility factors
    // For example: model configuration, sampling parameters, etc.
    // TODO: Implement more compatibility checks

    return true;
}

float Batch::compute_efficiency() const {
    if (requests_.empty()) {
        return 0.0f;
    }

    // Calculate the ratio of theoretical maximum throughput to actual throughput
    return 1.0f - stats_.padding_ratio;
}

std::chrono::microseconds Batch::processing_time() const {
    if (start_time_ == std::chrono::system_clock::time_point::min()) {
        return std::chrono::microseconds(0);
    }
    
    if (end_time_ == std::chrono::system_clock::time_point::min()) {
        return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now() - start_time_);
    }
    
    return std::chrono::duration_cast<std::chrono::microseconds>(
        end_time_ - start_time_);
}

} // namespace deeppowers 