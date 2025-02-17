#pragma once

#include "../request_queue/request.hpp"
#include <vector>
#include <memory>

namespace deeppowers {

// Batch status
enum class BatchStatus {
    PENDING,    // Waiting to be processed
    PROCESSING, // Being processed
    COMPLETED,  // Completed
    FAILED      // Failed
};

// Batch statistics
struct BatchStats {
    size_t total_tokens;           // Total token count
    size_t max_sequence_length;    // Maximum sequence length
    size_t min_sequence_length;    // Minimum sequence length
    float avg_sequence_length;     // Average sequence length
    size_t padding_tokens;         // Padding token count
    float padding_ratio;           // Padding ratio
};

// Batch instance
class Batch {
public:
    explicit Batch(const std::vector<RequestPtr>& requests);
    ~Batch() = default;

    // Basic attribute access
    const std::vector<RequestPtr>& requests() const { return requests_; }
    BatchStatus status() const { return status_; }
    const BatchStats& stats() const { return stats_; }
    
    // Status update
    void set_status(BatchStatus status);
    void mark_started();
    void mark_completed();
    void mark_failed(const std::string& error_message);

    // Batch operations
    void update_stats();  // Update statistics
    bool is_compatible(const RequestPtr& request) const;  // Check if the request is compatible with the current batch
    float compute_efficiency() const;  // Compute batch processing efficiency

    // Time related
    std::chrono::system_clock::time_point created_time() const { return created_time_; }
    std::chrono::system_clock::time_point start_time() const { return start_time_; }
    std::chrono::system_clock::time_point end_time() const { return end_time_; }
    std::chrono::microseconds processing_time() const;

private:
    std::vector<RequestPtr> requests_;  // Requests in the batch
    BatchStatus status_;                // Current status
    BatchStats stats_;                  // Statistics
    
    // Timestamp
    std::chrono::system_clock::time_point created_time_;  // Creation time
    std::chrono::system_clock::time_point start_time_;    // Start processing time
    std::chrono::system_clock::time_point end_time_;      // End time
};

using BatchPtr = std::shared_ptr<Batch>;

} // namespace deeppowers 