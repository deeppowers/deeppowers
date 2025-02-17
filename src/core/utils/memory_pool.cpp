#include "memory_pool.hpp"
#include <cstdlib>
#include <algorithm>

namespace deeppowers {

// Initialize global memory pool
MemoryPool* g_memory_pool = nullptr;

MemoryPool::MemoryPool(size_t block_size, size_t initial_blocks)
    : block_size_(block_size)
    , alignment_(std::max(sizeof(void*), static_cast<size_t>(16)))  // 16-byte alignment
    , free_list_(nullptr)
    , total_blocks_(0)
    , used_blocks_(0) {
    
    // Create initial blocks
    expand(initial_blocks);
}

MemoryPool::~MemoryPool() {
    // Free all allocated blocks
    for (auto block : all_blocks_) {
        std::free(block->data);
        delete block;
    }
}

void* MemoryPool::allocate(size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Round up size to alignment
    size = (size + alignment_ - 1) & ~(alignment_ - 1);
    
    // Find suitable block
    MemoryBlock* block = free_list_;
    MemoryBlock* prev = nullptr;
    
    while (block) {
        if (block->size >= size) {
            // Remove from free list
            if (prev) {
                prev->next = block->next;
            } else {
                free_list_ = block->next;
            }
            
            block->in_use = true;
            used_blocks_++;
            return block->data;
        }
        prev = block;
        block = block->next;
    }
    
    // No suitable block found, expand pool
    size_t new_blocks = std::max(size_t(1), (size + block_size_ - 1) / block_size_);
    expand(new_blocks);
    
    // Try allocation again
    return allocate(size);
}

void MemoryPool::deallocate(void* ptr) {
    if (!ptr) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Find block containing ptr
    auto it = std::find_if(all_blocks_.begin(), all_blocks_.end(),
                          [ptr](const MemoryBlock* block) {
                              return block->data == ptr;
                          });
    
    if (it != all_blocks_.end()) {
        MemoryBlock* block = *it;
        block->in_use = false;
        used_blocks_--;
        
        // Add to free list
        block->next = free_list_;
        free_list_ = block;
    }
}

MemoryBlock* MemoryPool::create_block(size_t size) {
    // Allocate memory for block
    void* data = std::aligned_alloc(alignment_, size);
    if (!data) {
        throw std::bad_alloc();
    }
    
    // Create block metadata
    auto* block = new MemoryBlock;
    block->data = data;
    block->size = size;
    block->in_use = false;
    block->next = nullptr;
    
    return block;
}

void MemoryPool::expand(size_t additional_blocks) {
    // Create new blocks
    for (size_t i = 0; i < additional_blocks; i++) {
        auto* block = create_block(block_size_);
        
        // Add to free list
        block->next = free_list_;
        free_list_ = block;
        
        // Add to all blocks list
        all_blocks_.push_back(block);
    }
    
    total_blocks_ += additional_blocks;
}

} // namespace deeppowers 