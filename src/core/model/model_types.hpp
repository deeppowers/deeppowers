#pragma once

#include <string>
#include <functional>
#include <memory>

namespace deeppowers {

class Model;

/**
 * Supported model formats
 */
enum class ModelFormat {
    AUTO,       // Auto-detect format
    ONNX,       // ONNX format
    PYTORCH,    // PyTorch format
    TENSORFLOW, // TensorFlow format
    CUSTOM      // Custom format
};

/**
 * Model loader function type
 */
using ModelLoaderFunc = std::function<std::shared_ptr<Model>(const std::string&)>;

/**
 * Model data types
 */
enum class DataType {
    FLOAT32,    // 32-bit floating point
    FLOAT16,    // 16-bit floating point
    INT8,       // 8-bit integer
    INT4,       // 4-bit integer
    UINT8,      // Unsigned 8-bit integer
    INT32       // 32-bit integer
};

/**
 * Model precision modes
 */
enum class PrecisionMode {
    FULL,       // Full precision (FP32)
    MIXED,      // Mixed precision (FP16/FP32)
    INT8,       // INT8 quantization
    INT4,       // INT4 quantization
    AUTO        // Automatic precision selection
};

/**
 * Model execution mode
 */
enum class ExecutionMode {
    SYNC,       // Synchronous execution
    ASYNC,      // Asynchronous execution
    STREAM      // Streaming execution
};

/**
 * Model optimization level
 */
enum class OptimizationLevel {
    NONE,       // No optimization
    O1,         // Basic optimizations
    O2,         // Medium optimizations
    O3          // Aggressive optimizations
};

} // namespace deeppowers 