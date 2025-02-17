#include "memory_pool.hpp"
#include <algorithm>
#include <cassert>

namespace deeppowers {

MemoryPool::MemoryPool(hal::Device* device, size_t initial_size, float growth_factor)
    : device_(device)
    , growth_factor_(growth_factor)
    , total_size_(0)
    , used_size_(0)
    , peak_used_(0) {
    
    if (!device_) {
        throw std::runtime_error("Device cannot be null");
    }

    if (growth_factor_ <= 1.0f) {
        throw std::runtime_error("Growth factor must be greater than 1.0");
    }

    // Initialize first chunk
    reserve(initial_size);
}

MemoryPool::~MemoryPool() {
    clear();
}

void* MemoryPool::allocate(size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Align size to 256 bytes
    size = (size + 255) & ~255;

    // Find best fit block
    Block* block = find_best_fit(size);

    // If no suitable block found, grow pool
    if (!block) {
        grow_pool(size);
        block = find_best_fit(size);
        if (!block) {
            throw std::runtime_error("Failed to allocate memory");
        }
    }

    // Split block if too large
    if (block->size > size + 1024) {  // Only split if difference is significant
        split_block(block, size);
    }

    // Mark block as used
    block->in_use = true;
    used_size_ += block->size;
    peak_used_ = std::max(peak_used_, used_size_);

    // Track allocation
    track_allocation(block->ptr, size);

    return block->ptr;
}

void MemoryPool::deallocate(void* ptr) {
    if (!ptr) return;

    std::lock_guard<std::mutex> lock(mutex_);

    // Find block
    auto it = ptr_to_block_.find(ptr);
    if (it == ptr_to_block_.end()) {
        throw std::runtime_error("Invalid pointer deallocation");
    }

    Block* block = it->second;
    block->in_use = false;
    used_size_ -= block->size;

    // Track deallocation
    track_deallocation(ptr);

    // Merge adjacent free blocks
    merge_free_blocks();

    // Defragment if fragmentation is high
    if (fragmentation_ratio() > 0.3f) {
        defragment();
    }
}

size_t MemoryPool::memory_usage() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return used_size_;
}

size_t MemoryPool::peak_memory_usage() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return peak_used_;
}

float MemoryPool::fragmentation_ratio() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (total_size_ == 0) return 0.0f;

    // Calculate total free space and number of free blocks
    size_t total_free = 0;
    size_t num_free_blocks = 0;
    
    for (const auto& chunk : chunks_) {
        for (const auto& block : chunk.blocks) {
            if (!block.in_use) {
                total_free += block.size;
                num_free_blocks++;
            }
        }
    }

    // Calculate average free block size
    float avg_free_size = num_free_blocks > 0 ? 
        static_cast<float>(total_free) / num_free_blocks : 0.0f;

    // Calculate fragmentation ratio based on average free block size
    return 1.0f - (avg_free_size / static_cast<float>(total_size_));
}

void MemoryPool::clear() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Free all chunks
    for (auto& chunk : chunks_) {
        device_->deallocate(chunk.ptr);
    }

    chunks_.clear();
    ptr_to_block_.clear();
    allocation_info_.clear();
    total_size_ = 0;
    used_size_ = 0;
    peak_used_ = 0;
}

void MemoryPool::reserve(size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Align size to 1MB
    size = (size + 1024*1024 - 1) & ~(1024*1024 - 1);

    // Allocate new chunk
    Chunk chunk;
    chunk.ptr = device_->allocate(size);
    chunk.size = size;
    chunk.used = 0;

    // Create initial free block
    Block block;
    block.ptr = chunk.ptr;
    block.size = size;
    block.offset = 0;
    block.in_use = false;
    block.pinned = false;

    chunk.blocks.push_back(block);
    ptr_to_block_[block.ptr] = &chunk.blocks.back();

    chunks_.push_back(chunk);
    total_size_ += size;
}

void MemoryPool::grow_pool(size_t min_size) {
    // Calculate new size
    size_t new_size = std::max(min_size, 
                              static_cast<size_t>(total_size_ * growth_factor_));
    
    // Reserve new memory
    reserve(new_size);
}

void MemoryPool::defragment() {
    // Skip if no fragmentation
    if (fragmentation_ratio() < 0.1f) return;

    // Create temporary storage for used blocks
    std::vector<std::pair<void*, size_t>> used_blocks;
    
    // Collect all used blocks
    for (const auto& chunk : chunks_) {
        for (const auto& block : chunk.blocks) {
            if (block.in_use) {
                used_blocks.emplace_back(block.ptr, block.size);
            }
        }
    }

    // Sort blocks by size
    std::sort(used_blocks.begin(), used_blocks.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    // Allocate new chunk
    size_t total_used = used_size_;
    Chunk new_chunk;
    new_chunk.ptr = device_->allocate(total_used);
    new_chunk.size = total_used;
    new_chunk.used = 0;

    // Copy used blocks to new location
    size_t offset = 0;
    for (const auto& [ptr, size] : used_blocks) {
        // Copy data
        device_->memcpy_to_device(
            static_cast<char*>(new_chunk.ptr) + offset,
            ptr,
            size);

        // Create new block
        Block block;
        block.ptr = static_cast<char*>(new_chunk.ptr) + offset;
        block.size = size;
        block.offset = offset;
        block.in_use = true;
        block.pinned = false;

        new_chunk.blocks.push_back(block);
        ptr_to_block_[block.ptr] = &new_chunk.blocks.back();

        offset += size;
    }

    // Free old chunks
    for (auto& chunk : chunks_) {
        device_->deallocate(chunk.ptr);
    }

    // Update state
    chunks_.clear();
    chunks_.push_back(new_chunk);
    total_size_ = total_used;
}

void MemoryPool::merge_free_blocks() {
    for (auto& chunk : chunks_) {
        auto it = chunk.blocks.begin();
        while (it != chunk.blocks.end()) {
            if (!it->in_use) {
                // Find next free block
                auto next = std::next(it);
                while (next != chunk.blocks.end() && !next->in_use) {
                    // Merge blocks
                    it->size += next->size;
                    ptr_to_block_.erase(next->ptr);
                    next = chunk.blocks.erase(next);
                }
            }
            ++it;
        }
    }
}

MemoryPool::Block* MemoryPool::find_best_fit(size_t size) {
    Block* best_fit = nullptr;
    size_t min_diff = std::numeric_limits<size_t>::max();

    for (auto& chunk : chunks_) {
        for (auto& block : chunk.blocks) {
            if (!block.in_use && block.size >= size) {
                size_t diff = block.size - size;
                if (diff < min_diff) {
                    min_diff = diff;
                    best_fit = &block;
                    if (diff == 0) break;  // Perfect fit
                }
            }
        }
        if (min_diff == 0) break;  // Perfect fit found
    }

    return best_fit;
}

void MemoryPool::split_block(Block* block, size_t size) {
    assert(block && block->size > size);

    // Create new block from remaining space
    Block new_block;
    new_block.ptr = static_cast<char*>(block->ptr) + size;
    new_block.size = block->size - size;
    new_block.offset = block->offset + size;
    new_block.in_use = false;
    new_block.pinned = false;

    // Update original block
    block->size = size;

    // Find chunk containing block
    for (auto& chunk : chunks_) {
        auto it = std::find_if(chunk.blocks.begin(), chunk.blocks.end(),
            [block](const Block& b) { return &b == block; });
        
        if (it != chunk.blocks.end()) {
            // Insert new block after current block
            chunk.blocks.insert(std::next(it), new_block);
            ptr_to_block_[new_block.ptr] = &*std::next(it);
            break;
        }
    }
}

void MemoryPool::update_statistics(void* ptr, size_t size) {
    auto& info = allocation_info_[ptr];
    info.size = size;
    info.frequency++;
    info.last_use = std::chrono::steady_clock::now();
}

void MemoryPool::track_allocation(void* ptr, size_t size) {
    update_statistics(ptr, size);
}

void MemoryPool::track_deallocation(void* ptr) {
    allocation_info_.erase(ptr);
}

} // namespace deeppowers