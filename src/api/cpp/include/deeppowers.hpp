#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace deeppowers {
namespace api {

// Forward declarations
class Model;
class GenerationConfig;
class GenerationResult;

// Generation configuration
struct GenerationConfig {
    std::string model_type = "gpt";         // Model type
    size_t max_tokens = 100;                // Maximum tokens to generate
    float temperature = 0.7f;               // Sampling temperature
    float top_p = 1.0f;                     // Nucleus sampling threshold
    float top_k = 0.0f;                     // Top-k sampling threshold
    std::vector<std::string> stop_tokens;   // Stop sequences
    bool stream = false;                    // Enable streaming generation
    size_t batch_size = 1;                 // Batch size for batch generation
};

// Generation result
struct GenerationResult {
    std::vector<std::string> texts;         // Generated texts
    std::vector<float> logprobs;           // Token log probabilities
    std::vector<std::vector<std::string>> tokens;  // Generated tokens
    std::vector<std::string> stop_reasons; // Reasons for stopping
    double generation_time;                // Generation time in seconds
};

// Streaming callback type
using StreamCallback = std::function<bool(const GenerationResult&)>;

// Model class
class Model {
public:
    // Constructor and destructor
    explicit Model(const std::string& model_path);
    ~Model();

    // Generation methods
    GenerationResult generate(const std::string& prompt,
                            const GenerationConfig& config = GenerationConfig());
    
    void generate_stream(const std::string& prompt,
                        StreamCallback callback,
                        const GenerationConfig& config = GenerationConfig());
    
    std::vector<GenerationResult> generate_batch(
        const std::vector<std::string>& prompts,
        const GenerationConfig& config = GenerationConfig());

    // Model information
    std::string model_type() const;
    std::string model_path() const;
    size_t vocab_size() const;
    size_t max_sequence_length() const;
    
    // Device management
    void to_device(const std::string& device);
    std::string device() const;
    
    // Configuration
    void set_config(const std::string& key, const std::string& value);
    std::string get_config(const std::string& key) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// Factory functions
std::shared_ptr<Model> load_model(const std::string& model_path);
std::vector<std::string> list_available_models();
bool is_model_available(const std::string& model_name);

// Version information
std::string version();
std::string cuda_version();
bool cuda_available();
size_t cuda_device_count();

} // namespace api
} // namespace deeppowers 