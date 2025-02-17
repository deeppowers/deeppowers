#pragma once

#include <string>
#include <string_view>
#include <unordered_set>
#include <mutex>

namespace deeppowers {

class StringPool {
public:
    // Get global instance
    static StringPool& instance();

    // Intern string
    std::string_view intern(const std::string& str);
    std::string_view intern(const char* str);
    std::string_view intern(std::string_view str);

    // Pool statistics
    size_t size() const;
    size_t memory_usage() const;

    // Clear pool
    void clear();

private:
    StringPool() = default;
    ~StringPool();

    // Disable copy/move
    StringPool(const StringPool&) = delete;
    StringPool& operator=(const StringPool&) = delete;
    StringPool(StringPool&&) = delete;
    StringPool& operator=(StringPool&&) = delete;

    // Internal implementation
    struct StringHash {
        using is_transparent = void;  // Enable heterogeneous lookup
        size_t operator()(const std::string& str) const {
            return std::hash<std::string>{}(str);
        }
        size_t operator()(const char* str) const {
            return std::hash<std::string_view>{}(str);
        }
        size_t operator()(std::string_view str) const {
            return std::hash<std::string_view>{}(str);
        }
    };

    struct StringEqual {
        using is_transparent = void;  // Enable heterogeneous lookup
        bool operator()(const std::string& lhs, const std::string& rhs) const {
            return lhs == rhs;
        }
        bool operator()(const std::string& lhs, const char* rhs) const {
            return lhs == rhs;
        }
        bool operator()(const std::string& lhs, std::string_view rhs) const {
            return lhs == rhs;
        }
    };

    // Storage
    std::unordered_set<std::string, StringHash, StringEqual> strings_;
    
    // Statistics
    size_t total_memory_;
    
    // Thread safety
    mutable std::mutex mutex_;
};

// Helper function to get string pool instance
inline StringPool& get_string_pool() {
    return StringPool::instance();
}

} // namespace deeppowers 