#pragma once

#include <string>
#include <memory>
#include "model.hpp"

namespace deeppowers {

/**
 * TensorFlow model loader class
 */
class TensorFlowLoader {
public:
    /**
     * Load TensorFlow model from file
     * @param path Path to TensorFlow model file or directory
     * @return Loaded model instance
     */
    static std::shared_ptr<Model> load(const std::string& path);
    
    /**
     * Save model to TensorFlow format
     * @param model Model to save
     * @param path Output file path
     */
    static void save(const std::shared_ptr<Model>& model, const std::string& path);
    
    /**
     * Check if file is a valid TensorFlow model
     * @param path Path to file or directory
     * @return True if file is a valid TensorFlow model
     */
    static bool is_valid(const std::string& path);
};

} // namespace deeppowers 