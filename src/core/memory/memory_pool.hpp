#pragma once

#include "../hal/hal.hpp"
#include <memory>
#include <vector>
#include <mutex>
#include <unordered_map>

namespace deeppowers {

/**
 * @brief Memory pool for efficient memory management
 * Provides memory allocation, deallocation, and reuse mechanisms
 */
class MemoryPool {
public:
    /**
     * @brief Constructor
     * @param device Device to allocate memory on
     * @param initial_size Initial pool size in bytes
     * @param growth_factor Factor to grow pool when out of memory
     */
    explicit MemoryPool(hal::Device* device,
                       size_t initial_size = 1024 * 1024 * 1024,  // 1GB
                       float growth_factor = 1.5f);
    ~MemoryPool();

    /**
     * @brief Allocate memory from the pool
     * @param size Size in bytes to allocate
     * @return Pointer to allocated memory
     */
    void* allocate(size_t size);

    /**
     * @brief Free memory back to the pool
     * @param ptr Pointer to memory to free
     */
    void deallocate(void* ptr);

    /**
     * @brief Get current memory usage
     * @return Current memory usage in bytes
     */
    size_t memory_usage() const;

    /**
     * @brief Get peak memory usage
     * @return Peak memory usage in bytes
     */
    size_t peak_memory_usage() const;

    /**
     * @brief Get fragmentation ratio
     * @return Fragmentation ratio (0-1)
     */
    float fragmentation_ratio() const;

    /**
     * @brief Clear all allocations and reset pool
     */
    void clear();

    /**
     * @brief Reserve memory in the pool
     * @param size Size in bytes to reserve
     */
    void reserve(size_t size);

private:
    // Memory block structure
    struct Block {
        void* ptr;          // Pointer to memory
        size_t size;        // Size of block
        size_t offset;      // Offset in pool
        bool in_use;        // Whether block is in use
        bool pinned;        // Whether block is pinned
    };

    // Memory chunk structure
    struct Chunk {
        void* ptr;          // Base pointer
        size_t size;        // Total size
        size_t used;        // Used size
        std::vector<Block> blocks;  // Memory blocks
    };

    // Internal helper methods
    void grow_pool(size_t min_size);
    void defragment();
    void merge_free_blocks();
    Block* find_best_fit(size_t size);
    void split_block(Block* block, size_t size);

    // Member variables
    hal::Device* device_;
    float growth_factor_;
    size_t total_size_;
    size_t used_size_;
    size_t peak_used_;
    std::vector<Chunk> chunks_;
    std::unordered_map<void*, Block*> ptr_to_block_;
    mutable std::mutex mutex_;

    // Memory tracking
    struct AllocationInfo {
        size_t size;
        size_t frequency;
        std::chrono::steady_clock::time_point last_use;
    };
    std::unordered_map<void*, AllocationInfo> allocation_info_;

    // Statistics
    void update_statistics(void* ptr, size_t size);
    void track_allocation(void* ptr, size_t size);
    void track_deallocation(void* ptr);
};
} 