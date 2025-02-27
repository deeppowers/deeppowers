#include "model_loader.hpp"
#include <fstream>
#include <algorithm>
#include <stdexcept>
#include <filesystem>
#include <iostream>

namespace deeppowers {

// Initialize static format handlers map
std::unordered_map<ModelFormat, ModelLoaderFunc> ModelLoader::format_handlers_;

std::shared_ptr<Model> ModelLoader::load(const std::string& path, ModelFormat format) {
    // Auto-detect format if not specified
    if (format == ModelFormat::AUTO) {
        format = detect_format(path);
    }
    
    // Check if format handler exists
    auto it = format_handlers_.find(format);
    if (it == format_handlers_.end()) {
        throw std::runtime_error("Unsupported model format: " + std::to_string(static_cast<int>(format)));
    }
    
    // Call format handler to load model
    try {
        return it->second(path);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load model: " + std::string(e.what()));
    }
}

void ModelLoader::convert(const std::string& input_path,
                         const std::string& output_path,
                         ModelFormat input_format,
                         ModelFormat output_format) {
    // Auto-detect input format if not specified
    if (input_format == ModelFormat::AUTO) {
        input_format = detect_format(input_path);
    }
    
    // Check if input format handler exists
    auto input_handler = format_handlers_.find(input_format);
    if (input_handler == format_handlers_.end()) {
        throw std::runtime_error("Unsupported input model format: " + 
                                std::to_string(static_cast<int>(input_format)));
    }
    
    // Check if output format handler exists
    auto output_handler = format_handlers_.find(output_format);
    if (output_handler == format_handlers_.end()) {
        throw std::runtime_error("Unsupported output model format: " + 
                                std::to_string(static_cast<int>(output_format)));
    }
    
    // Load model using input format handler
    std::shared_ptr<Model> model;
    try {
        model = input_handler->second(input_path);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load model for conversion: " + std::string(e.what()));
    }
    
    // Save model in output format
    try {
        model->save(output_path, output_format);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to save converted model: " + std::string(e.what()));
    }
}

void ModelLoader::register_format(ModelFormat format, ModelLoaderFunc loader) {
    format_handlers_[format] = std::move(loader);
}

ModelFormat ModelLoader::detect_format(const std::string& path) {
    // Check if file exists
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("Model file not found: " + path);
    }
    
    // Check if it's a directory
    if (std::filesystem::is_directory(path)) {
        // Check for PyTorch format
        if (std::filesystem::exists(path + "/model.pt") || 
            std::filesystem::exists(path + "/pytorch_model.bin")) {
            return ModelFormat::PYTORCH;
        }
        
        // Check for TensorFlow format
        if (std::filesystem::exists(path + "/saved_model.pb")) {
            return ModelFormat::TENSORFLOW;
        }
        
        // Default to custom format for directories
        return ModelFormat::CUSTOM;
    }
    
    // Check file extension
    std::string extension = std::filesystem::path(path).extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    
    if (extension == ".onnx") {
        return ModelFormat::ONNX;
    } else if (extension == ".pt" || extension == ".pth") {
        return ModelFormat::PYTORCH;
    } else if (extension == ".pb" || extension == ".savedmodel") {
        return ModelFormat::TENSORFLOW;
    }
    
    // Try to detect by reading file header
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open model file: " + path);
    }
    
    // Read first 8 bytes
    char header[8];
    file.read(header, sizeof(header));
    
    // ONNX files start with "ONNX"
    if (header[0] == 'O' && header[1] == 'N' && header[2] == 'N' && header[3] == 'X') {
        return ModelFormat::ONNX;
    }
    
    // PyTorch files start with "PK\x03\x04"
    if (header[0] == 'P' && header[1] == 'K' && header[2] == 0x03 && header[3] == 0x04) {
        return ModelFormat::PYTORCH;
    }
    
    // Default to custom format
    return ModelFormat::CUSTOM;
}

// Register default format handlers
namespace {
    
// ONNX model loader
std::shared_ptr<Model> load_onnx(const std::string& path) {
    // TODO: Implement ONNX model loading
    std::cout << "Loading ONNX model from: " << path << std::endl;
    throw std::runtime_error("ONNX model loading not implemented yet");
}

// PyTorch model loader
std::shared_ptr<Model> load_pytorch(const std::string& path) {
    // TODO: Implement PyTorch model loading
    std::cout << "Loading PyTorch model from: " << path << std::endl;
    throw std::runtime_error("PyTorch model loading not implemented yet");
}

// TensorFlow model loader
std::shared_ptr<Model> load_tensorflow(const std::string& path) {
    // TODO: Implement TensorFlow model loading
    std::cout << "Loading TensorFlow model from: " << path << std::endl;
    throw std::runtime_error("TensorFlow model loading not implemented yet");
}

// Register default handlers
struct RegisterDefaultHandlers {
    RegisterDefaultHandlers() {
        ModelLoader::register_format(ModelFormat::ONNX, load_onnx);
        ModelLoader::register_format(ModelFormat::PYTORCH, load_pytorch);
        ModelLoader::register_format(ModelFormat::TENSORFLOW, load_tensorflow);
    }
} register_default_handlers;

} // anonymous namespace

} // namespace deeppowers 