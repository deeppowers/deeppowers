#pragma once

#include <vector>
#include <memory>
#include <mutex>

namespace deeppowers {

// Memory block structure
struct MemoryBlock {
    void* data;
    size_t size;
    bool in_use;
    MemoryBlock* next;
};

class MemoryPool {
public:
    // Initialize pool with block size and initial capacity
    MemoryPool(size_t block_size, size_t initial_blocks = 1024);
    ~MemoryPool();

    // Allocate memory from pool
    void* allocate(size_t size);
    
    // Return memory to pool
    void deallocate(void* ptr);

    // Pool statistics
    size_t get_total_blocks() const { return total_blocks_; }
    size_t get_used_blocks() const { return used_blocks_; }
    size_t get_block_size() const { return block_size_; }

private:
    // Create new block
    MemoryBlock* create_block(size_t size);
    
    // Expand pool
    void expand(size_t additional_blocks = 1024);

    // Pool configuration
    const size_t block_size_;
    const size_t alignment_;
    
    // Pool state
    MemoryBlock* free_list_;
    std::vector<MemoryBlock*> all_blocks_;
    size_t total_blocks_;
    size_t used_blocks_;
    
    // Thread safety
    mutable std::mutex mutex_;
};

// Global memory pool instance
extern MemoryPool* g_memory_pool;

// Smart pointer for pool allocation
template<typename T>
class PoolPtr {
public:
    template<typename... Args>
    static PoolPtr<T> make(Args&&... args) {
        void* ptr = g_memory_pool->allocate(sizeof(T));
        T* obj = new(ptr) T(std::forward<Args>(args)...);
        return PoolPtr<T>(obj);
    }

    ~PoolPtr() {
        if (ptr_) {
            ptr_->~T();
            g_memory_pool->deallocate(ptr_);
        }
    }

    // Move operations
    PoolPtr(PoolPtr&& other) noexcept : ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }
    
    PoolPtr& operator=(PoolPtr&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    // Disable copy operations
    PoolPtr(const PoolPtr&) = delete;
    PoolPtr& operator=(const PoolPtr&) = delete;

    // Access operators
    T* operator->() const { return ptr_; }
    T& operator*() const { return *ptr_; }
    T* get() const { return ptr_; }

    // Reset pointer
    void reset() {
        if (ptr_) {
            ptr_->~T();
            g_memory_pool->deallocate(ptr_);
            ptr_ = nullptr;
        }
    }

private:
    explicit PoolPtr(T* ptr) : ptr_(ptr) {}
    T* ptr_;
};

} // namespace deeppowers 