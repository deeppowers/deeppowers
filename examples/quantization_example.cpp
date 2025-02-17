#include <deeppowers.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>

using namespace deeppowers::api;
using namespace std::chrono;

// Helper function: Generate calibration data
std::vector<std::string> load_calibration_data(const std::string& file_path, size_t num_samples) {
    std::vector<std::string> data;
    std::ifstream file(file_path);
    std::string line;
    
    while (std::getline(file, line) && data.size() < num_samples) {
        if (!line.empty()) {
            data.push_back(line);
        }
    }
    
    return data;
}

// Helper function: Evaluate generation quality
float evaluate_quality(const std::string& reference, const std::string& generated) {
    // Here we use simple character matching rate as an example
    // In practice, use more complex evaluation metrics (e.g., BLEU, ROUGE)
    size_t matches = 0;
    size_t total = std::max(reference.length(), generated.length());
    
    for (size_t i = 0; i < std::min(reference.length(), generated.length()); i++) {
        if (reference[i] == generated[i]) {
            matches++;
        }
    }
    
    return static_cast<float>(matches) / total;
}

int main() {
    try {
        // Initialize model
        std::cout << "Loading model..." << std::endl;
        auto model = load_model("gpt2");
        
        // Record original model size
        size_t original_size = model->get_model_size();
        std::cout << "Original model size: " << original_size / (1024 * 1024) 
                  << " MB" << std::endl;
        
        // Load calibration data
        std::cout << "Loading calibration data..." << std::endl;
        auto calibration_data = load_calibration_data("data/calibration.txt", 100);
        if (calibration_data.empty()) {
            throw std::runtime_error("Failed to load calibration data");
        }
        
        // Generate baseline results
        std::cout << "Generating baseline results..." << std::endl;
        GenerationConfig config;
        config.max_tokens = 50;
        config.temperature = 0.7f;
        
        const std::string test_prompt = "The quick brown fox jumps over";
        auto baseline_result = model->generate(test_prompt, config);
        
        // Test different quantization configurations
        std::vector<std::pair<std::string, QuantizationType>> quant_configs = {
            {"INT8", QuantizationType::INT8},
            {"INT4", QuantizationType::INT4},
            {"MIXED", QuantizationType::MIXED}
        };
        
        for (const auto& [name, type] : quant_configs) {
            std::cout << "\nTesting " << name << " quantization:" << std::endl;
            
            // Configure quantization parameters
            QuantizationConfig quant_config;
            quant_config.type = type;
            quant_config.method = QuantizationMethod::POST_TRAINING;
            quant_config.per_channel = true;
            
            if (type == QuantizationType::MIXED) {
                // Set different layer precisions
                std::unordered_map<std::string, QuantizationType> layer_precisions;
                layer_precisions["attention"] = QuantizationType::INT8;
                layer_precisions["ffn"] = QuantizationType::INT4;
                layer_precisions["embedding"] = QuantizationType::NONE;
                quant_config.layer_precisions = layer_precisions;
            }
            
            // Apply quantization
            model->set_quantization_config(quant_config);
            model->quantize(calibration_data);
            
            // Record quantized model size
            size_t quantized_size = model->get_model_size();
            float compression_ratio = static_cast<float>(original_size) / quantized_size;
            
            std::cout << "Quantized model size: " << quantized_size / (1024 * 1024) 
                      << " MB (compression ratio: " << compression_ratio << "x)" << std::endl;
            
            // Test generation performance
            auto start_time = high_resolution_clock::now();
            auto quantized_result = model->generate(test_prompt, config);
            auto end_time = high_resolution_clock::now();
            
            auto generation_time = duration_cast<milliseconds>(
                end_time - start_time).count() / 1000.0;
            
            // Evaluate generation quality
            float quality_score = evaluate_quality(baseline_result.texts[0], 
                                                quantized_result.texts[0]);
            
            // Output results
            std::cout << "Generation time: " << generation_time << " seconds" << std::endl;
            std::cout << "Quality score: " << quality_score << std::endl;
            std::cout << "Generated text: " << quantized_result.texts[0] << std::endl;
            
            // Restore to original model for next test
            model->dequantize();
        }
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
} 