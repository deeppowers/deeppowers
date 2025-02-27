#pragma once

#include <string>
#include <memory>
#include "model.hpp"

namespace deeppowers {

/**
 * PyTorch model loader class
 */
class PyTorchLoader {
public:
    /**
     * Load PyTorch model from file
     * @param path Path to PyTorch model file
     * @return Loaded model instance
     */
    static std::shared_ptr<Model> load(const std::string& path);
    
    /**
     * Save model to PyTorch format
     * @param model Model to save
     * @param path Output file path
     */
    static void save(const std::shared_ptr<Model>& model, const std::string& path);
    
    /**
     * Check if file is a valid PyTorch model
     * @param path Path to file
     * @return True if file is a valid PyTorch model
     */
    static bool is_valid(const std::string& path);
};

} // namespace deeppowers 