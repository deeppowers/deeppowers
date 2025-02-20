#pragma once

#include <string>
#include <vector>

namespace deeppowers {
namespace core {

class Tokenizer {
public:
    Tokenizer();
    ~Tokenizer();

    std::vector<int> encode(const std::string& text);
    std::string decode(const std::vector<int>& tokens);
};

} // namespace core
} // namespace deeppowers 