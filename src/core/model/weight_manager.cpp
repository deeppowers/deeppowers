#include "weight_manager.hpp"
#include "../memory/memory_pool.hpp"
#include <fstream>
#include <nlohmann/json.hpp>
#include <algorithm>

namespace deeppowers {

using json = nlohmann::json;

WeightManager::WeightManager(hal::Device* device, MemoryPool* memory_pool)
    : device_(device)
    , memory_pool_(memory_pool)
    , total_memory_(0)
    , is_optimized_(false) {
    
    if (!device_) {
        throw std::runtime_error("Device cannot be null");
    }
}

WeightManager::~WeightManager() {
    clear();
}

void WeightManager::load_weights(const std::string& path, const std::string& format) {
    // Clear existing weights
    clear();

    // Load weights based on format
    if (format == "safetensors") {
        parse_safetensors(path);
    } else if (format == "bin") {
        parse_binary(path);
    } else {
        throw std::runtime_error("Unsupported weight format: " + format);
    }

    // Validate and optimize weights
    validate_weights();
    optimize_memory_layout();
}

void WeightManager::parse_safetensors(const std::string& path) {
    // Open file
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open weight file: " + path);
    }

    // Read header
    uint64_t header_size;
    file.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));

    // Read metadata
    std::string metadata_str(header_size, '\0');
    file.read(&metadata_str[0], header_size);
    auto metadata = json::parse(metadata_str);

    // Read tensors
    for (const auto& [name, info] : metadata["tensors"].items()) {
        // Parse tensor metadata
        std::vector<int64_t> shape = info["shape"];
        std::string dtype_str = info["dtype"];
        uint64_t data_offset = info["data_offsets"][0];
        uint64_t data_size = info["data_offsets"][1] - data_offset;

        // Convert dtype string to enum
        hal::DataType dtype;
        if (dtype_str == "F32") {
            dtype = hal::DataType::FLOAT32;
        } else if (dtype_str == "F16") {
            dtype = hal::DataType::FLOAT16;
        } else {
            throw std::runtime_error("Unsupported dtype: " + dtype_str);
        }

        // Create tensor
        WeightInfo weight_info;
        weight_info.shape = shape;
        weight_info.dtype = dtype;
        weight_info.offset = data_offset;
        weight_info.pinned = false;

        // Allocate memory using memory pool if available
        void* data_ptr;
        if (memory_pool_) {
            data_ptr = memory_pool_->allocate(data_size);
        } else {
            data_ptr = device_->allocate(data_size);
        }

        // Create tensor
        weight_info.tensor = std::make_unique<hal::Tensor>(shape, dtype, device_);

        // Read data
        file.seekg(data_offset + header_size);
        std::vector<uint8_t> host_data(data_size);
        file.read(reinterpret_cast<char*>(host_data.data()), data_size);

        // Copy data to device
        device_->memcpy_to_device(data_ptr, host_data.data(), data_size);

        // Store weight info
        weights_[name] = std::move(weight_info);
        total_memory_ += data_size;
    }
}

void WeightManager::parse_binary(const std::string& path) {
    // Implementation for binary format
    // TODO: Implement binary format parsing
}

void WeightManager::save_weights(const std::string& path, const std::string& format) {
    if (format == "safetensors") {
        // Create metadata
        json metadata;
        metadata["tensors"] = json::object();

        // Calculate total data size
        size_t total_data_size = 0;
        for (const auto& [name, info] : weights_) {
            size_t tensor_size = info.tensor->size_in_bytes();
            metadata["tensors"][name] = {
                {"dtype", info.dtype == hal::DataType::FLOAT32 ? "F32" : "F16"},
                {"shape", info.shape},
                {"data_offsets", {total_data_size, total_data_size + tensor_size}}
            };
            total_data_size += tensor_size;
        }

        // Write file
        std::ofstream file(path, std::ios::binary);
        
        // Write header size
        std::string metadata_str = metadata.dump();
        uint64_t header_size = metadata_str.size();
        file.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
        
        // Write metadata
        file.write(metadata_str.data(), metadata_str.size());
        
        // Write tensor data
        for (const auto& [name, info] : weights_) {
            std::vector<uint8_t> host_data(info.tensor->size_in_bytes());
            device_->memcpy_to_host(host_data.data(), info.tensor->data(), 
                                  info.tensor->size_in_bytes());
            file.write(reinterpret_cast<const char*>(host_data.data()), 
                      host_data.size());
        }
    } else {
        throw std::runtime_error("Unsupported save format: " + format);
    }
}

hal::Tensor* WeightManager::get_tensor(const std::string& name) {
    auto it = weights_.find(name);
    if (it == weights_.end()) {
        throw std::runtime_error("Weight not found: " + name);
    }
    return it->second.tensor.get();
}

const hal::Tensor* WeightManager::get_tensor(const std::string& name) const {
    auto it = weights_.find(name);
    if (it == weights_.end()) {
        throw std::runtime_error("Weight not found: " + name);
    }
    return it->second.tensor.get();
}

void WeightManager::to_device(hal::Device* device) {
    if (!device) {
        throw std::runtime_error("Device cannot be null");
    }

    // Move each tensor to new device
    for (auto& [name, info] : weights_) {
        void* new_data = device->allocate(info.tensor->size_in_bytes());
        device->memcpy_to_device(new_data, info.tensor->data(), 
                               info.tensor->size_in_bytes());
        info.tensor = std::make_unique<hal::Tensor>(info.shape, info.dtype, device);
    }

    device_ = device;
    is_optimized_ = false;
}

size_t WeightManager::total_size() const {
    return total_memory_;
}

size_t WeightManager::memory_usage() const {
    size_t usage = 0;
    for (const auto& [name, info] : weights_) {
        usage += info.tensor->size_in_bytes();
    }
    return usage;
}

void WeightManager::clear() {
    weights_.clear();
    total_memory_ = 0;
    is_optimized_ = false;
}

void WeightManager::optimize_memory_layout() {
    if (is_optimized_) {
        return;
    }

    // Perform memory optimizations
    defragment_memory();
    pin_frequently_used_weights();
    merge_small_weights();

    is_optimized_ = true;
}

void WeightManager::validate_weights() {
    // Validate each tensor
    for (const auto& [name, info] : weights_) {
        // Check shape validity
        if (info.shape.empty()) {
            throw std::runtime_error("Invalid shape for weight: " + name);
        }

        // Check data type
        if (info.dtype != hal::DataType::FLOAT32 && 
            info.dtype != hal::DataType::FLOAT16) {
            throw std::runtime_error("Invalid data type for weight: " + name);
        }

        // Validate tensor data
        if (!info.tensor || !info.tensor->data()) {
            throw std::runtime_error("Invalid tensor data for weight: " + name);
        }
    }
}

void WeightManager::defragment_memory() {
    // Sort weights by size
    std::vector<std::pair<std::string, size_t>> weight_sizes;
    for (const auto& [name, info] : weights_) {
        weight_sizes.emplace_back(name, info.tensor->size_in_bytes());
    }
    std::sort(weight_sizes.begin(), weight_sizes.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    // Reallocate weights in order
    size_t current_offset = 0;
    for (const auto& [name, size] : weight_sizes) {
        auto& info = weights_[name];
        info.offset = current_offset;
        current_offset += size;
    }
}

void WeightManager::pin_frequently_used_weights() {
    // Pin weights that are frequently accessed
    // TODO: Implement weight pinning based on access patterns
}

void WeightManager::merge_small_weights() {
    // Merge small weights into a single buffer
    // TODO: Implement small weight merging
}

} // namespace deeppowers 