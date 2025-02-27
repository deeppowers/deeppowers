#pragma once

#include <vector>
#include <memory>
#include <string>
#include "model_types.hpp"

namespace deeppowers {

/**
 * Tensor class for multi-dimensional data
 */
class Tensor {
public:
    /**
     * Create a tensor with specified shape and data type
     * @param shape Tensor dimensions
     * @param dtype Data type
     */
    Tensor(const std::vector<size_t>& shape, DataType dtype);
    
    /**
     * Create a tensor with specified shape, data type, and data
     * @param shape Tensor dimensions
     * @param dtype Data type
     * @param data Pointer to data
     * @param copy_data Whether to copy data or take ownership
     */
    Tensor(const std::vector<size_t>& shape, DataType dtype, void* data, bool copy_data = true);
    
    /**
     * Copy constructor
     */
    Tensor(const Tensor& other);
    
    /**
     * Move constructor
     */
    Tensor(Tensor&& other) noexcept;
    
    /**
     * Copy assignment operator
     */
    Tensor& operator=(const Tensor& other);
    
    /**
     * Move assignment operator
     */
    Tensor& operator=(Tensor&& other) noexcept;
    
    /**
     * Destructor
     */
    ~Tensor();
    
    /**
     * Get tensor shape
     * @return Vector of dimensions
     */
    const std::vector<size_t>& shape() const;
    
    /**
     * Get tensor data type
     * @return Data type
     */
    DataType dtype() const;
    
    /**
     * Get total number of elements
     * @return Element count
     */
    size_t size() const;
    
    /**
     * Get raw data pointer
     * @return Void pointer to data
     */
    void* data();
    
    /**
     * Get raw data pointer (const version)
     * @return Const void pointer to data
     */
    const void* data() const;
    
    /**
     * Get typed data pointer
     * @return Typed pointer to data
     */
    template<typename T>
    T* data() {
        return static_cast<T*>(data());
    }
    
    /**
     * Get typed data pointer (const version)
     * @return Const typed pointer to data
     */
    template<typename T>
    const T* data() const {
        return static_cast<const T*>(data());
    }
    
    /**
     * Reshape tensor (without changing data)
     * @param new_shape New dimensions
     * @return Reference to this tensor
     */
    Tensor& reshape(const std::vector<size_t>& new_shape);
    
    /**
     * Convert tensor to different data type
     * @param dtype Target data type
     * @return New tensor with converted data
     */
    Tensor to(DataType dtype) const;
    
    /**
     * Move tensor to device
     * @param device_name Target device
     * @return Reference to this tensor
     */
    Tensor& to_device(const std::string& device_name);
    
    /**
     * Get tensor device
     * @return Device name
     */
    std::string device() const;
    
    /**
     * Create a copy of this tensor
     * @return New tensor with copied data
     */
    Tensor clone() const;
    
    /**
     * Get string representation of tensor
     * @return String representation
     */
    std::string to_string() const;

private:
    std::vector<size_t> shape_;
    DataType dtype_;
    std::shared_ptr<void> data_;
    size_t size_;
    std::string device_;
    
    // Calculate total size from shape
    size_t calculate_size(const std::vector<size_t>& shape);
    
    // Allocate memory for tensor data
    void allocate();
    
    // Copy data from another tensor
    void copy_from(const Tensor& other);
};

} // namespace deeppowers 