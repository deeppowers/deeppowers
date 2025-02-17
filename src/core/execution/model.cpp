#include "model.hpp"
#include <stdexcept>

namespace deeppowers {

// Initialize static member
std::unordered_map<std::string,
    std::function<std::unique_ptr<Model>(
        const ModelConfig&, hal::Device*)>> ModelFactory::model_creators_;

Model::Model(const ModelConfig& config, hal::Device* device)
    : config_(config)
    , device_(device) {
    
    if (!device_) {
        throw std::runtime_error("Device cannot be null");
    }
}

std::unique_ptr<Model> ModelFactory::create_model(
    const std::string& model_name,
    const ModelConfig& config,
    hal::Device* device) {
    
    auto it = model_creators_.find(model_name);
    if (it == model_creators_.end()) {
        throw std::runtime_error("Unknown model type: " + model_name);
    }
    
    return it->second(config, device);
}

void ModelFactory::register_model_creator(
    const std::string& model_name,
    std::function<std::unique_ptr<Model>(
        const ModelConfig&, hal::Device*)> creator) {
    
    if (model_creators_.find(model_name) != model_creators_.end()) {
        throw std::runtime_error("Model type already registered: " + model_name);
    }
    
    model_creators_[model_name] = std::move(creator);
}

} // namespace deeppowers