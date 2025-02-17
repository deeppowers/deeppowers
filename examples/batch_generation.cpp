#include <deeppowers.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

using namespace deeppowers::api;
using namespace std::chrono;

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
        
        // Prepare batch of prompts
        std::vector<std::string> prompts = {
            "Write a story about",
            "Explain the concept of",
            "What is the meaning of",
            "How does the process of"
        };
        
        // Set batch size
        config.batch_size = prompts.size();
        
        // Generate texts
        std::cout << "Generating " << prompts.size() << " texts in parallel..." << std::endl;
        auto start_time = high_resolution_clock::now();
        
        auto results = model->generate_batch(prompts, config);
        
        auto end_time = high_resolution_clock::now();
        auto total_time = duration_cast<milliseconds>(end_time - start_time).count() / 1000.0;
        
        // Output results
        std::cout << "\nGenerated texts:" << std::endl;
        for (size_t i = 0; i < results.size(); i++) {
            std::cout << "\nPrompt " << i + 1 << ": " << prompts[i] << std::endl;
            std::cout << "Generated: " << results[i].texts[0] << std::endl;
            std::cout << "Generation time: " << results[i].generation_time << " seconds" << std::endl;
        }
        
        // Output performance statistics
        std::cout << "\nPerformance statistics:" << std::endl;
        std::cout << "Total time: " << total_time << " seconds" << std::endl;
        std::cout << "Average time per text: " << total_time / prompts.size() 
                  << " seconds" << std::endl;
        std::cout << "Throughput: " << prompts.size() / total_time 
                  << " texts per second" << std::endl;
        
        // Calculate batch processing speedup
        std::cout << "\nBatch processing speedup:" << std::endl;
        
        // Compare with sequential generation
        config.batch_size = 1;
        start_time = high_resolution_clock::now();
        
        for (const auto& prompt : prompts) {
            model->generate(prompt, config);
        }
        
        end_time = high_resolution_clock::now();
        auto sequential_time = duration_cast<milliseconds>(end_time - start_time).count() / 1000.0;
        
        std::cout << "Sequential processing time: " << sequential_time << " seconds" << std::endl;
        std::cout << "Speedup factor: " << sequential_time / total_time << "x" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
} 