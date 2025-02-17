#pragma once

#include <string>
#include <stdexcept>

namespace deeppowers {
namespace common {

// Error codes
enum class ErrorCode {
    SUCCESS = 0,
    
    // System errors
    SYSTEM_ERROR = 1000,
    OUT_OF_MEMORY = 1001,
    DEVICE_ERROR = 1002,
    
    // Argument errors
    INVALID_ARGUMENT = 2000,
    INVALID_CONFIG = 2001,
    INVALID_STATE = 2002,
    
    // Runtime errors
    RUNTIME_ERROR = 3000,
    CUDA_ERROR = 3001,
    TIMEOUT = 3002,
    
    // Graph compilation errors
    GRAPH_ERROR = 4000,
    CYCLE_DETECTED = 4001,
    INVALID_NODE = 4002,
    
    // Memory errors
    MEMORY_ERROR = 5000,
    ALLOCATION_FAILED = 5001,
    BUFFER_OVERFLOW = 5002,
    
    // Other errors
    UNKNOWN_ERROR = 9999
};

// Base exception class
class Exception : public std::runtime_error {
public:
    Exception(ErrorCode code, const std::string& message)
        : std::runtime_error(message)
        , code_(code) {}
    
    ErrorCode code() const { return code_; }
    
private:
    ErrorCode code_;
};

// System exception
class SystemException : public Exception {
public:
    SystemException(const std::string& message)
        : Exception(ErrorCode::SYSTEM_ERROR, message) {}
};

// Device exception
class DeviceException : public Exception {
public:
    DeviceException(const std::string& message)
        : Exception(ErrorCode::DEVICE_ERROR, message) {}
};

// Argument exception
class ArgumentException : public Exception {
public:
    ArgumentException(const std::string& message)
        : Exception(ErrorCode::INVALID_ARGUMENT, message) {}
};

// Configuration exception
class ConfigException : public Exception {
public:
    ConfigException(const std::string& message)
        : Exception(ErrorCode::INVALID_CONFIG, message) {}
};

// Runtime exception
class RuntimeException : public Exception {
public:
    RuntimeException(const std::string& message)
        : Exception(ErrorCode::RUNTIME_ERROR, message) {}
};

// CUDA exception
class CUDAException : public Exception {
public:
    CUDAException(const std::string& message)
        : Exception(ErrorCode::CUDA_ERROR, message) {}
};

// Graph compilation exception
class GraphException : public Exception {
public:
    GraphException(const std::string& message)
        : Exception(ErrorCode::GRAPH_ERROR, message) {}
};

// Memory exception
class MemoryException : public Exception {
public:
    MemoryException(const std::string& message)
        : Exception(ErrorCode::MEMORY_ERROR, message) {}
};

// Error handling macros
#define CHECK_CUDA(err) { \
    cudaError_t cuda_err = (err); \
    if (cuda_err != cudaSuccess) { \
        throw CUDAException(std::string("CUDA error: ") + cudaGetErrorString(cuda_err)); \
    } \
}

#define CHECK_ARG(condition, message) { \
    if (!(condition)) { \
        throw ArgumentException(message); \
    } \
}

#define CHECK_STATE(condition, message) { \
    if (!(condition)) { \
        throw RuntimeException(message); \
    } \
}

} // namespace common
} // namespace deeppowers 