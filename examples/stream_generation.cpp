#include <deeppowers.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include <atomic>

using namespace deeppowers::api;

// Atomic variable to control generation process
std::atomic<bool> stop_generation{false};

// Handle Ctrl+C signal
void signal_handler(int signal) {
    stop_generation = true;
}

int main() {
    try {
        // Set signal handling
        std::signal(SIGINT, signal_handler);
        
        // Initialize model
        std::cout << "Loading model..." << std::endl;
        auto model = load_model("gpt2");
        
        // Set generation configuration
        GenerationConfig config;
        config.max_tokens = 200;
        config.temperature = 0.7f;
        config.top_p = 0.9f;
        config.stream = true;  // Enable streaming generation
        
        // Prepare prompt text
        std::string prompt;
        std::cout << "Enter your prompt (Ctrl+C to stop generation): ";
        std::getline(std::cin, prompt);
        
        // Streaming generation callback function
        size_t total_tokens = 0;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        auto stream_callback = [&](const GenerationResult& chunk) {
            if (stop_generation) {
                std::cout << "\nGeneration stopped by user." << std::endl;
                return false;
            }
            
            // Print newly generated text
            std::cout << chunk.texts[0] << std::flush;
            
            // Update statistics
            if (chunk.logprobs) {
                total_tokens += chunk.texts[0].length();
            }
            
            // Simulate typing effect
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            
            return true;  // Continue generation
        };
        
        // Start streaming generation
        std::cout << "\nGenerating (streaming mode)..." << std::endl;
        model->generate_stream(prompt, stream_callback, config);
        
        // Calculate total generation time
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count() / 1000.0;
        
        // Output statistics
        std::cout << "\n\nGeneration statistics:" << std::endl;
        std::cout << "Total tokens generated: " << total_tokens << std::endl;
        std::cout << "Total time: " << total_time << " seconds" << std::endl;
        std::cout << "Generation speed: " << total_tokens / total_time 
                  << " tokens per second" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
} 