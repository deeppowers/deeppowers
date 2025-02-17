#include <deeppowers.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <future>

using namespace deeppowers::api;

// Async generation example
void async_generation_example(Model* model) {
    std::cout << "\nAsync Generation Example:" << std::endl;
    
    GenerationConfig config;
    config.max_tokens = 50;
    
    // Create multiple async generation tasks
    std::vector<std::future<GenerationResult>> futures;
    std::vector<std::string> prompts = {
        "Write a story about",
        "Explain the concept of",
        "Describe the process of"
    };
    
    for (const auto& prompt : prompts) {
        futures.push_back(std::async(std::launch::async, [model, prompt, config]() {
            return model->generate(prompt, config);
        }));
    }
    
    // Wait and process results
    for (size_t i = 0; i < futures.size(); i++) {
        auto result = futures[i].get();
        std::cout << "Result " << i + 1 << ": " << result.texts[0] << std::endl;
    }
}

// Custom stop condition example
void custom_stop_condition_example(Model* model) {
    std::cout << "\nCustom Stop Condition Example:" << std::endl;
    
    GenerationConfig config;
    config.max_tokens = 100;
    config.stop_tokens = {"\n\n", "END", "."};  // Multiple stop tokens
    
    auto result = model->generate("Write a short sentence", config);
    std::cout << "Generated text: " << result.texts[0] << std::endl;
    if (result.stop_reasons) {
        std::cout << "Stop reason: " << result.stop_reasons->front() << std::endl;
    }
}

// Interactive generation example
void interactive_generation_example(Model* model) {
    std::cout << "\nInteractive Generation Example:" << std::endl;
    
    GenerationConfig config;
    config.max_tokens = 30;
    config.stream = true;
    
    std::string conversation;
    std::string user_input;
    
    while (true) {
        std::cout << "\nYou: ";
        std::getline(std::cin, user_input);
        
        if (user_input == "quit" || user_input == "exit") {
            break;
        }
        
        // Update conversation history
        conversation += "User: " + user_input + "\nAssistant: ";
        
        // Streaming generation reply
        std::cout << "Assistant: ";
        model->generate_stream(conversation, 
            [](const GenerationResult& chunk) {
                std::cout << chunk.texts[0] << std::flush;
                return true;
            },
            config);
        std::cout << std::endl;
        
        // Update conversation history
        auto result = model->generate(conversation, config);
        conversation += result.texts[0] + "\n";
    }
}

// Conditional generation example
void conditional_generation_example(Model* model) {
    std::cout << "\nConditional Generation Example:" << std::endl;
    
    GenerationConfig config;
    config.max_tokens = 50;
    
    // Generate with different control tokens
    std::vector<std::pair<std::string, std::string>> control_tokens = {
        {"[POSITIVE]", "Write about a happy event"},
        {"[NEGATIVE]", "Write about a sad event"},
        {"[NEUTRAL]", "Write about a normal day"}
    };
    
    for (const auto& [token, prompt] : control_tokens) {
        std::cout << "\nGenerating with control token: " << token << std::endl;
        auto result = model->generate(token + " " + prompt, config);
        std::cout << "Generated text: " << result.texts[0] << std::endl;
    }
}

// Long text processing example
void long_text_processing_example(Model* model) {
    std::cout << "\nLong Text Processing Example:" << std::endl;
    
    // Create a long text
    std::string long_text(model->max_sequence_length() - 100, 'x');
    
    GenerationConfig config;
    config.max_tokens = 50;
    
    // Use sliding window to process long text
    size_t window_size = 1024;
    size_t stride = 512;
    
    for (size_t i = 0; i < long_text.length(); i += stride) {
        size_t window_length = std::min(window_size, long_text.length() - i);
        std::string window = long_text.substr(i, window_length);
        
        auto result = model->generate(window, config);
        std::cout << "Window " << (i / stride + 1) << " generation: " 
                  << result.texts[0] << std::endl;
    }
}

// Model ensemble example
void model_ensemble_example() {
    std::cout << "\nModel Ensemble Example:" << std::endl;
    
    // Load multiple models
    auto model1 = load_model("gpt2");
    auto model2 = load_model("gpt2-medium");
    
    GenerationConfig config;
    config.max_tokens = 50;
    
    const std::string prompt = "The future of AI is";
    
    // Generate text from each model
    auto result1 = model1->generate(prompt, config);
    auto result2 = model2->generate(prompt, config);
    
    // Simple integration strategy: select the longest generation result
    const std::string& final_result = 
        (result1.texts[0].length() > result2.texts[0].length()) ?
        result1.texts[0] : result2.texts[0];
    
    std::cout << "Ensemble result: " << final_result << std::endl;
}

// Custom sampling strategy example
void custom_sampling_example(Model* model) {
    std::cout << "\nCustom Sampling Strategy Example:" << std::endl;
    
    GenerationConfig config;
    config.max_tokens = 50;
    
    // Test different sampling strategies
    std::vector<std::tuple<float, float, float>> sampling_params = {
        {0.0f, 1.0f, 0.0f},   // Greedy
        {0.7f, 0.9f, 0.0f},   // Nucleus
        {1.0f, 1.0f, 50.0f},  // Top-k
        {0.7f, 0.9f, 50.0f}   // Hybrid
    };
    
    const std::string prompt = "Once upon a time";
    
    for (const auto& [temp, top_p, top_k] : sampling_params) {
        config.temperature = temp;
        config.top_p = top_p;
        config.top_k = top_k;
        
        auto result = model->generate(prompt, config);
        
        std::cout << "\nSampling strategy (temp=" << temp 
                  << ", top_p=" << top_p 
                  << ", top_k=" << top_k << "):" << std::endl;
        std::cout << "Generated text: " << result.texts[0] << std::endl;
    }
}

int main() {
    try {
        // Initialize model
        std::cout << "Loading model..." << std::endl;
        auto model = load_model("gpt2");
        
        // Run various advanced features examples
        async_generation_example(model.get());
        custom_stop_condition_example(model.get());
        interactive_generation_example(model.get());
        conditional_generation_example(model.get());
        long_text_processing_example(model.get());
        model_ensemble_example();
        custom_sampling_example(model.get());
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
} 