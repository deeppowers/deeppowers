#include "string_pool.hpp"
#include <cassert>

namespace deeppowers {

StringPool& StringPool::instance() {
    static StringPool instance;
    return instance;
}

StringPool::~StringPool() {
    clear();
}

std::string_view StringPool::intern(const std::string& str) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Try to find existing string
    auto it = strings_.find(str);
    if (it != strings_.end()) {
        return std::string_view(*it);
    }
    
    // Insert new string
    auto [new_it, inserted] = strings_.insert(str);
    assert(inserted);
    
    // Update memory usage
    total_memory_ += str.capacity() + sizeof(std::string);
    
    return std::string_view(*new_it);
}

std::string_view StringPool::intern(const char* str) {
    return intern(std::string_view(str));
}

std::string_view StringPool::intern(std::string_view str) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Try to find existing string
    auto it = strings_.find(str);
    if (it != strings_.end()) {
        return std::string_view(*it);
    }
    
    // Insert new string
    auto [new_it, inserted] = strings_.insert(std::string(str));
    assert(inserted);
    
    // Update memory usage
    total_memory_ += str.length() + 1 + sizeof(std::string);
    
    return std::string_view(*new_it);
}

size_t StringPool::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return strings_.size();
}

size_t StringPool::memory_usage() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return total_memory_;
}

void StringPool::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    strings_.clear();
    total_memory_ = 0;
}

} // namespace deeppowers 