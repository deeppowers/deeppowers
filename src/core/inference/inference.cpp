#include "inference/inference.hpp"

namespace deeppowers {
namespace core {

InferenceEngine::InferenceEngine() {}

InferenceEngine::~InferenceEngine() {}

std::vector<float> InferenceEngine::run(const std::vector<int>& input_ids) {
    // TODO: Implement actual inference
    return std::vector<float>();
}

} // namespace core
} // namespace deeppowers 