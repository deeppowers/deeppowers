#include <deeppowers.hpp>
#include <iostream>
#include <string>

using namespace deeppowers::api;

int main() {
    try {
        // Initialize model
        std::cout << "Loading model..." << std::endl;
        auto model = load_model("gpt2");
        
        // Set generation configuration
        GenerationConfig config;
        config.max_tokens = 50;
        config.temperature = 0.7f;
        config.top_p = 0.9f;
        
        // Prepare prompt text
        std::string prompt;
        std::cout << "Enter your prompt: ";
        std::getline(std::cin, prompt);
        
        // Generate text
        std::cout << "Generating..." << std::endl;
        auto result = model->generate(prompt, config);
        
        // Output results
        std::cout << "\nGenerated text:" << std::endl;
        std::cout << result.texts[0] << std::endl;
        
        // Output generation statistics
        std::cout << "\nGeneration statistics:" << std::endl;
        std::cout << "Time taken: " << result.generation_time << " seconds" << std::endl;
        if (result.logprobs) {
            std::cout << "Average token probability: " 
                      << std::exp(std::accumulate(result.logprobs->begin(),
                                                result.logprobs->end(), 0.0f) 
                                 / result.logprobs->size())
                      << std::endl;
        }
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
} 