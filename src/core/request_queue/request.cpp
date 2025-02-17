#include "request.hpp"

namespace deeppowers {

Request::Request(const std::string& request_id,
                const std::string& prompt,
                RequestPriority priority)
    : id_(request_id)
    , prompt_(prompt)
    , status_(RequestStatus::PENDING)
    , priority_(priority)
    , created_time_(std::chrono::system_clock::now())
    , start_time_(std::chrono::system_clock::time_point::min())
    , end_time_(std::chrono::system_clock::time_point::min()) {
}

void Request::set_status(RequestStatus status) {
    status_ = status;
    
    // Update timestamp based on status
    auto now = std::chrono::system_clock::now();
    switch (status) {
        case RequestStatus::RUNNING:
            start_time_ = now;
            break;
        case RequestStatus::COMPLETED:
        case RequestStatus::FAILED:
            end_time_ = now;
            break;
        default:
            break;
    }
}

void Request::mark_started() {
    set_status(RequestStatus::RUNNING);
}

void Request::mark_completed() {
    set_status(RequestStatus::COMPLETED);
    result_.processing_time = processing_time();
}

void Request::mark_failed(const std::string& error_message) {
    set_status(RequestStatus::FAILED);
    result_.error_message = error_message;
    result_.processing_time = processing_time();
}

std::chrono::microseconds Request::wait_time() const {
    if (start_time_ == std::chrono::system_clock::time_point::min()) {
        // If not started yet, calculate wait time from creation to now
        return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now() - created_time_);
    } else {
        // Otherwise, calculate wait time from creation to start time
        return std::chrono::duration_cast<std::chrono::microseconds>(
            start_time_ - created_time_);
    }
}

std::chrono::microseconds Request::processing_time() const {
    if (start_time_ == std::chrono::system_clock::time_point::min()) {
        // If not started yet
        return std::chrono::microseconds(0);
    }
    
    if (end_time_ == std::chrono::system_clock::time_point::min()) {
        // If not finished yet
        return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now() - start_time_);
    } else {
        // Otherwise, calculate processing time from start to end
        return std::chrono::duration_cast<std::chrono::microseconds>(
            end_time_ - start_time_);
    }
}

} // namespace deeppowers 