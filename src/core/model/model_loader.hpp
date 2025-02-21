#pragma once

#include <string>
#include <memory>
#include <unordered_map>
#include "model_types.hpp"

namespace deeppowers {

/**
 * Model loader class supporting various model formats
 */
class ModelLoader {
public:
    /**
     * Load model from file
     * @param path Path to model file
     * @param format Model format (default: auto-detect)
     * @return Loaded model instance
     */
    static std::shared_ptr<Model> load(const std::string& path,
                                     ModelFormat format = ModelFormat::AUTO);
    
    /**
     * Convert model between formats
     * @param input_path Input model path
     * @param output_path Output model path
     * @param input_format Input model format
     * @param output_format Output model format
     */
    static void convert(const std::string& input_path,
                       const std::string& output_path,
                       ModelFormat input_format,
                       ModelFormat output_format);
    
    /**
     * Register custom model format handler
     * @param format Model format
     * @param loader Loader function
     */
    static void register_format(ModelFormat format,
                              ModelLoaderFunc loader);

private:
    // Format detection
    static ModelFormat detect_format(const std::string& path);
    
    // Format handlers
    static std::unordered_map<ModelFormat, ModelLoaderFunc> format_handlers_;
};

} // namespace deeppowers 