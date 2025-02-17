#pragma once

#include "../hal/hal.hpp"
#include <string>
#include <unordered_map>
#include <memory>
#include <vector>

namespace deeppowers {

// Forward declarations
class MemoryPool;

/**
 * @brief Weight management system for deep learning models
 * Handles loading, storing, and managing model weights efficiently
 */
class WeightManager {
public:
    explicit WeightManager(hal::Device* device, MemoryPool* memory_pool = nullptr);
    ~WeightManager();

    /**
     * @brief Load weights from a file
     * @param path Path to the weight file
     * @param format Format of the weight file (e.g., "safetensors", "bin")
     */
    void load_weights(const std::string& path, const std::string& format = "safetensors");

    /**
     * @brief Save weights to a file
     * @param path Path to save the weights
     * @param format Format to save the weights in
     */
    void save_weights(const std::string& path, const std::string& format = "safetensors");

    /**
     * @brief Get a tensor by name
     * @param name Name of the tensor
     * @return Pointer to the tensor
     */
    hal::Tensor* get_tensor(const std::string& name);

    /**
     * @brief Get a tensor by name (const version)
     * @param name Name of the tensor
     * @return Const pointer to the tensor
     */
    const hal::Tensor* get_tensor(const std::string& name) const;

    /**
     * @brief Move weights to a specific device
     * @param device Target device
     */
    void to_device(hal::Device* device);

    /**
     * @brief Get the total size of all weights in bytes
     * @return Total size in bytes
     */
    size_t total_size() const;

    /**
     * @brief Get memory usage statistics
     * @return Memory usage in bytes
     */
    size_t memory_usage() const;

    /**
     * @brief Clear all weights and free memory
     */
    void clear();

private:
    // Internal helper methods
    void parse_safetensors(const std::string& path);
    void parse_binary(const std::string& path);
    void optimize_memory_layout();
    void validate_weights();

    // Weight storage
    struct WeightInfo {
        std::unique_ptr<hal::Tensor> tensor;
        std::vector<int64_t> shape;
        hal::DataType dtype;
        size_t offset;
        bool pinned;
    };

    // Member variables
    hal::Device* device_;
    MemoryPool* memory_pool_;
    std::unordered_map<std::string, WeightInfo> weights_;
    size_t total_memory_;
    bool is_optimized_;

    // Memory optimization
    void defragment_memory();
    void pin_frequently_used_weights();
    void merge_small_weights();
}; 