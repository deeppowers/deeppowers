#include "tensor.hpp"
#include <numeric>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <iomanip>

namespace deeppowers {

Tensor::Tensor(const std::vector<size_t>& shape, DataType dtype)
    : shape_(shape)
    , dtype_(dtype)
    , data_(nullptr)
    , size_(calculate_size(shape))
    , device_("cpu") {
    allocate();
}

Tensor::Tensor(const std::vector<size_t>& shape, DataType dtype, void* data, bool copy_data)
    : shape_(shape)
    , dtype_(dtype)
    , data_(nullptr)
    , size_(calculate_size(shape))
    , device_("cpu") {
    
    if (copy_data) {
        allocate();
        size_t bytes = size_ * get_dtype_size(dtype_);
        std::memcpy(data_.get(), data, bytes);
    } else {
        // Take ownership of data
        data_ = std::shared_ptr<void>(data, [](void* p) { free(p); });
    }
}

Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_)
    , dtype_(other.dtype_)
    , data_(nullptr)
    , size_(other.size_)
    , device_(other.device_) {
    copy_from(other);
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_))
    , dtype_(other.dtype_)
    , data_(std::move(other.data_))
    , size_(other.size_)
    , device_(std::move(other.device_)) {
    other.size_ = 0;
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        shape_ = other.shape_;
        dtype_ = other.dtype_;
        size_ = other.size_;
        device_ = other.device_;
        copy_from(other);
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        shape_ = std::move(other.shape_);
        dtype_ = other.dtype_;
        data_ = std::move(other.data_);
        size_ = other.size_;
        device_ = std::move(other.device_);
        other.size_ = 0;
    }
    return *this;
}

Tensor::~Tensor() {
    // Shared_ptr will handle memory deallocation
}

const std::vector<size_t>& Tensor::shape() const {
    return shape_;
}

DataType Tensor::dtype() const {
    return dtype_;
}

size_t Tensor::size() const {
    return size_;
}

void* Tensor::data() {
    return data_.get();
}

const void* Tensor::data() const {
    return data_.get();
}

Tensor& Tensor::reshape(const std::vector<size_t>& new_shape) {
    size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), 
                                     1, std::multiplies<size_t>());
    
    if (new_size != size_) {
        throw std::runtime_error("Reshape operation cannot change the total number of elements");
    }
    
    shape_ = new_shape;
    return *this;
}

Tensor Tensor::to(DataType dtype) const {
    if (dtype == dtype_) {
        return clone();
    }
    
    Tensor result(shape_, dtype);
    
    // Implement type conversion logic here
    // This is a simplified version that only handles some conversions
    
    if (dtype_ == DataType::FLOAT32 && dtype == DataType::FLOAT16) {
        // Float32 to Float16 conversion
        const float* src = data<float>();
        uint16_t* dst = result.data<uint16_t>();
        
        for (size_t i = 0; i < size_; ++i) {
            // Simple float32 to float16 conversion (not IEEE compliant)
            dst[i] = convert_float32_to_float16(src[i]);
        }
    } else if (dtype_ == DataType::FLOAT16 && dtype == DataType::FLOAT32) {
        // Float16 to Float32 conversion
        const uint16_t* src = data<uint16_t>();
        float* dst = result.data<float>();
        
        for (size_t i = 0; i < size_; ++i) {
            // Simple float16 to float32 conversion (not IEEE compliant)
            dst[i] = convert_float16_to_float32(src[i]);
        }
    } else if (dtype_ == DataType::FLOAT32 && dtype == DataType::INT8) {
        // Float32 to Int8 conversion
        const float* src = data<float>();
        int8_t* dst = result.data<int8_t>();
        
        for (size_t i = 0; i < size_; ++i) {
            // Simple quantization
            dst[i] = static_cast<int8_t>(std::max(std::min(src[i] * 127.0f, 127.0f), -128.0f));
        }
    } else if (dtype_ == DataType::INT8 && dtype == DataType::FLOAT32) {
        // Int8 to Float32 conversion
        const int8_t* src = data<int8_t>();
        float* dst = result.data<float>();
        
        for (size_t i = 0; i < size_; ++i) {
            // Simple dequantization
            dst[i] = static_cast<float>(src[i]) / 127.0f;
        }
    } else {
        throw std::runtime_error("Unsupported data type conversion");
    }
    
    return result;
}

Tensor& Tensor::to_device(const std::string& device_name) {
    if (device_ == device_name) {
        return *this;
    }
    
    // TODO: Implement device transfer logic
    // For now, we only support CPU
    if (device_name != "cpu") {
        throw std::runtime_error("Only CPU device is currently supported");
    }
    
    device_ = device_name;
    return *this;
}

std::string Tensor::device() const {
    return device_;
}

Tensor Tensor::clone() const {
    Tensor copy(shape_, dtype_);
    size_t bytes = size_ * get_dtype_size(dtype_);
    std::memcpy(copy.data_.get(), data_.get(), bytes);
    copy.device_ = device_;
    return copy;
}

std::string Tensor::to_string() const {
    std::ostringstream oss;
    oss << "Tensor(shape=[";
    
    for (size_t i = 0; i < shape_.size(); ++i) {
        oss << shape_[i];
        if (i < shape_.size() - 1) {
            oss << ", ";
        }
    }
    
    oss << "], dtype=";
    
    switch (dtype_) {
        case DataType::FLOAT32:
            oss << "float32";
            break;
        case DataType::FLOAT16:
            oss << "float16";
            break;
        case DataType::INT8:
            oss << "int8";
            break;
        case DataType::INT4:
            oss << "int4";
            break;
        case DataType::UINT8:
            oss << "uint8";
            break;
        case DataType::INT32:
            oss << "int32";
            break;
        default:
            oss << "unknown";
    }
    
    oss << ", device=" << device_ << ")";
    
    // Print a preview of the data if tensor is small
    if (size_ <= 10) {
        oss << " [";
        
        if (dtype_ == DataType::FLOAT32) {
            const float* data_ptr = data<float>();
            for (size_t i = 0; i < size_; ++i) {
                oss << std::fixed << std::setprecision(4) << data_ptr[i];
                if (i < size_ - 1) {
                    oss << ", ";
                }
            }
        } else if (dtype_ == DataType::INT8) {
            const int8_t* data_ptr = data<int8_t>();
            for (size_t i = 0; i < size_; ++i) {
                oss << static_cast<int>(data_ptr[i]);
                if (i < size_ - 1) {
                    oss << ", ";
                }
            }
        }
        // Add more data type handling as needed
        
        oss << "]";
    }
    
    return oss.str();
}

size_t Tensor::calculate_size(const std::vector<size_t>& shape) {
    return std::accumulate(shape.begin(), shape.end(), 
                          1, std::multiplies<size_t>());
}

void Tensor::allocate() {
    size_t bytes = size_ * get_dtype_size(dtype_);
    data_ = std::shared_ptr<void>(malloc(bytes), free);
    
    if (!data_) {
        throw std::runtime_error("Failed to allocate memory for tensor");
    }
    
    // Initialize to zero
    std::memset(data_.get(), 0, bytes);
}

void Tensor::copy_from(const Tensor& other) {
    allocate();
    size_t bytes = size_ * get_dtype_size(dtype_);
    std::memcpy(data_.get(), other.data_.get(), bytes);
}

// Helper functions for data type conversions

uint16_t convert_float32_to_float16(float value) {
    // Simplified float32 to float16 conversion
    // This is not IEEE compliant, just for demonstration
    uint32_t x = *reinterpret_cast<uint32_t*>(&value);
    uint16_t h = ((x >> 16) & 0x8000) |
                ((((x >> 23) - 127 + 15) & 0x1F) << 10) |
                ((x >> 13) & 0x3FF);
    return h;
}

float convert_float16_to_float32(uint16_t value) {
    // Simplified float16 to float32 conversion
    // This is not IEEE compliant, just for demonstration
    uint32_t x = ((value & 0x8000) << 16) |
                (((((value >> 10) & 0x1F) - 15 + 127) & 0xFF) << 23) |
                ((value & 0x3FF) << 13);
    return *reinterpret_cast<float*>(&x);
}

size_t get_dtype_size(DataType dtype) {
    switch (dtype) {
        case DataType::FLOAT32:
            return 4;
        case DataType::FLOAT16:
            return 2;
        case DataType::INT8:
        case DataType::UINT8:
            return 1;
        case DataType::INT4:
            return 1; // Packed, 2 values per byte
        case DataType::INT32:
            return 4;
        default:
            throw std::runtime_error("Unknown data type");
    }
}

} // namespace deeppowers 