#pragma once

#include <vector>

namespace deeppowers {
namespace core {

class InferenceEngine {
public:
    InferenceEngine();
    ~InferenceEngine();

    std::vector<float> run(const std::vector<int>& input_ids);
};

} // namespace core
} // namespace deeppowers 