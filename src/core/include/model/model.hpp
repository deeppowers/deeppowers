#pragma once

#include <vector>

namespace deeppowers {
namespace core {

class Model {
public:
    Model();
    ~Model();

    std::vector<float> forward(const std::vector<int>& input_ids);
};

} // namespace core
} // namespace deeppowers 