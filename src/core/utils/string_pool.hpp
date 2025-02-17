#pragma once

#include <string>
#include <string_view>
#include <unordered_set>
#include <memory>
#include <mutex>

namespace deeppowers {

// String pool for memory optimization
class StringPool {
public:
    // Constructor
    StringPool() = default;
    
    // Destructor
    ~StringPool() = default;
    
    // Delete copy constructor and assignment
    StringPool(const StringPool&) = delete;
    StringPool& operator=(const StringPool&) = delete;
    
    // Intern a string
    std::string_view intern(const std::string& str);
    std::string_view intern(std::string_view str);
    
    // Get number of strings in pool
    size_t size() const;
    
    // Clear the pool
    void clear();

private:
    // Set of interned strings
    std::unordered_set<std::string> strings_;
    
    // Mutex for thread safety
    mutable std::mutex mutex_;
};

} // namespace deeppowers 