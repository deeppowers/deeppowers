#include "../src/core/tokenizer/tokenizer.hpp"
#include <iostream>
#include <chrono>
#include <fstream>
#include <random>
#include <cassert>

using namespace deeppowers;
using namespace std::chrono;

// Helper function to generate random text
std::string generate_random_text(size_t length) {
    static const char charset[] = 
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789 ";
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, sizeof(charset) - 2);
    
    std::string text;
    text.reserve(length);
    for (size_t i = 0; i < length; ++i) {
        text += charset[dis(gen)];
    }
    return text;
}

// Helper function to measure execution time
template<typename Func>
double measure_time(Func&& func) {
    auto start = high_resolution_clock::now();
    func();
    auto end = high_resolution_clock::now();
    return duration_cast<milliseconds>(end - start).count() / 1000.0;
}

// Test basic tokenization functionality
void test_basic_functionality() {
    std::cout << "Testing basic tokenization functionality..." << std::endl;
    
    Tokenizer tokenizer(TokenizerType::BPE);
    
    // Train tokenizer on sample data
    std::vector<std::string> training_texts = {
        "Hello world",
        "This is a test",
        "Testing tokenizer functionality",
        "GPU acceleration is amazing"
    };
    
    tokenizer.train(training_texts, 100);
    
    // Test single text tokenization
    std::string test_text = "Hello world, this is a test";
    auto tokens = tokenizer.encode(test_text);
    auto decoded = tokenizer.decode(tokens);
    
    std::cout << "Original: " << test_text << std::endl;
    std::cout << "Decoded: " << decoded << std::endl;
    
    assert(decoded.length() > 0);
    std::cout << "Basic functionality test passed." << std::endl;
}

// Test GPU acceleration
void test_gpu_acceleration() {
    std::cout << "\nTesting GPU acceleration..." << std::endl;
    
    Tokenizer tokenizer(TokenizerType::BPE);
    
    // Check GPU availability
    if (!tokenizer.is_gpu_available()) {
        std::cout << "GPU is not available, skipping GPU tests." << std::endl;
        return;
    }
    
    // Generate test data
    const size_t num_texts = 1000;
    const size_t text_length = 1000;
    std::vector<std::string> texts;
    texts.reserve(num_texts);
    for (size_t i = 0; i < num_texts; ++i) {
        texts.push_back(generate_random_text(text_length));
    }
    
    // Train tokenizer
    tokenizer.train(texts, 1000);
    
    // Convert texts to string_view
    std::vector<std::string_view> text_views;
    text_views.reserve(texts.size());
    for (const auto& text : texts) {
        text_views.push_back(text);
    }
    
    // Test CPU batch processing
    double cpu_time = measure_time([&]() {
        tokenizer.set_device_type(DeviceType::CPU);
        auto cpu_results = tokenizer.encode_batch(text_views, true);
    });
    
    // Test GPU batch processing
    double gpu_time = measure_time([&]() {
        tokenizer.set_device_type(DeviceType::GPU);
        auto gpu_results = tokenizer.encode_batch_gpu(text_views, true);
    });
    
    std::cout << "CPU processing time: " << cpu_time << "s" << std::endl;
    std::cout << "GPU processing time: " << gpu_time << "s" << std::endl;
    std::cout << "Speedup: " << cpu_time / gpu_time << "x" << std::endl;
}

// Test memory management
void test_memory_management() {
    std::cout << "\nTesting memory management..." << std::endl;
    
    Tokenizer tokenizer(TokenizerType::BPE);
    
    // Generate large test data
    const size_t num_texts = 10000;
    const size_t text_length = 1000;
    std::vector<std::string> texts;
    texts.reserve(num_texts);
    for (size_t i = 0; i < num_texts; ++i) {
        texts.push_back(generate_random_text(text_length));
    }
    
    // Train tokenizer
    tokenizer.train(texts, 1000);
    
    // Test GPU memory management
    if (tokenizer.is_gpu_available()) {
        std::cout << "Testing GPU memory management..." << std::endl;
        
        // Switch to GPU mode multiple times
        for (int i = 0; i < 5; ++i) {
            tokenizer.set_device_type(DeviceType::GPU);
            std::vector<std::string_view> text_views(texts.begin(), texts.end());
            auto results = tokenizer.encode_batch_gpu(text_views, true);
            tokenizer.set_device_type(DeviceType::CPU);
        }
    }
    
    std::cout << "Memory management test passed." << std::endl;
}

// Test error handling
void test_error_handling() {
    std::cout << "\nTesting error handling..." << std::endl;
    
    Tokenizer tokenizer(TokenizerType::BPE);
    
    // Test invalid device type
    try {
        tokenizer.set_device_type(DeviceType::GPU);
        if (!tokenizer.is_gpu_available()) {
            std::cout << "Correctly caught GPU unavailability." << std::endl;
        }
    } catch (const std::runtime_error& e) {
        std::cout << "Correctly caught error: " << e.what() << std::endl;
    }
    
    // Test invalid text input
    try {
        std::string empty_text;
        auto tokens = tokenizer.encode(empty_text);
        std::cout << "Empty text handling passed." << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Error handling empty text: " << e.what() << std::endl;
    }
}

// Test batch processing with different sizes
void test_batch_processing() {
    std::cout << "\nTesting batch processing..." << std::endl;
    
    Tokenizer tokenizer(TokenizerType::BPE);
    
    // Generate test data with different batch sizes
    std::vector<size_t> batch_sizes = {1, 10, 100, 1000};
    
    for (size_t batch_size : batch_sizes) {
        std::cout << "Testing batch size: " << batch_size << std::endl;
        
        // Generate test data
        std::vector<std::string> texts;
        texts.reserve(batch_size);
        for (size_t i = 0; i < batch_size; ++i) {
            texts.push_back(generate_random_text(100));
        }
        
        // Train tokenizer
        tokenizer.train(texts, 1000);
        
        std::vector<std::string_view> text_views(texts.begin(), texts.end());
        
        // Test CPU batch processing
        double cpu_time = measure_time([&]() {
            tokenizer.set_device_type(DeviceType::CPU);
            auto cpu_results = tokenizer.encode_batch(text_views, true);
        });
        
        // Test GPU batch processing if available
        if (tokenizer.is_gpu_available()) {
            double gpu_time = measure_time([&]() {
                tokenizer.set_device_type(DeviceType::GPU);
                auto gpu_results = tokenizer.encode_batch_gpu(text_views, true);
            });
            
            std::cout << "  CPU time: " << cpu_time << "s" << std::endl;
            std::cout << "  GPU time: " << gpu_time << "s" << std::endl;
            std::cout << "  Speedup: " << cpu_time / gpu_time << "x" << std::endl;
        }
    }
}

int main() {
    try {
        test_basic_functionality();
        test_gpu_acceleration();
        test_memory_management();
        test_error_handling();
        test_batch_processing();
        
        std::cout << "\nAll tests completed successfully!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
        return 1;
    }
} 