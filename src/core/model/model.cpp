#include "model.hpp"
#include <stdexcept>

namespace deeppowers {

Model::Model() {
    // Default constructor
}

Model::~Model() {
    // Virtual destructor
}

Tensor Model::forward(const Tensor& input) {
    // Base implementation - should be overridden by derived classes
    throw std::runtime_error("forward() not implemented in base Model class");
}

std::vector<Tensor> Model::forward_batch(const std::vector<Tensor>& inputs) {
    // Default implementation processes each input individually
    std::vector<Tensor> outputs;
    outputs.reserve(inputs.size());
    
    for (const auto& input : inputs) {
        outputs.push_back(forward(input));
    }
    
    return outputs;
}

void Model::save(const std::string& path, ModelFormat format) {
    // Base implementation - should be overridden by derived classes
    throw std::runtime_error("save() not implemented in base Model class");
}

std::unordered_map<std::string, std::string> Model::config() const {
    // Base implementation returns empty config
    return {};
}

std::string Model::device() const {
    // Base implementation returns "cpu"
    return "cpu";
}

void Model::to(const std::string& device_name) {
    // Base implementation - should be overridden by derived classes
    throw std::runtime_error("to() not implemented in base Model class");
}

std::string Model::model_type() const {
    // Base implementation returns "unknown"
    return "unknown";
}

PrecisionMode Model::precision() const {
    // Base implementation returns FULL precision
    return PrecisionMode::FULL;
}

void Model::set_precision(PrecisionMode precision) {
    // Base implementation - should be overridden by derived classes
    throw std::runtime_error("set_precision() not implemented in base Model class");
}

OptimizationLevel Model::optimization_level() const {
    // Base implementation returns NONE
    return OptimizationLevel::NONE;
}

void Model::set_optimization_level(OptimizationLevel level) {
    // Base implementation - should be overridden by derived classes
    throw std::runtime_error("set_optimization_level() not implemented in base Model class");
}

} // namespace deeppowers 