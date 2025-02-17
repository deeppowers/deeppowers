#include <gtest/gtest.h>
#include <deeppowers.hpp>
#include <memory>
#include <chrono>
#include <thread>

using namespace deeppowers::api;

class GenerationTest : public ::testing::Test {
protected:
    void SetUp() override {
        model = std::make_shared<Model>("test_model");
        config = GenerationConfig();
        config.max_tokens = 20;
        config.temperature = 0.7f;
    }

    std::shared_ptr<Model> model;
    GenerationConfig config;
};

// Basic generation functionality test
TEST_F(GenerationTest, BasicGeneration) {
    const std::string prompt = "Hello, how are you?";
    auto result = model->generate(prompt, config);
    
    EXPECT_FALSE(result.texts.empty());
    EXPECT_GT(result.texts[0].length(), prompt.length());
    EXPECT_GT(result.generation_time, 0.0);
    
    if (result.logprobs) {
        EXPECT_EQ(result.logprobs->size(), result.texts[0].length() - prompt.length());
    }
}

// Batch generation test
TEST_F(GenerationTest, BatchGeneration) {
    std::vector<std::string> prompts = {
        "Hello world",
        "The weather is",
        "OpenAI is"
    };
    config.batch_size = prompts.size();
    
    auto results = model->generate_batch(prompts, config);
    
    EXPECT_EQ(results.size(), prompts.size());
    for (size_t i = 0; i < results.size(); i++) {
        EXPECT_FALSE(results[i].texts.empty());
        EXPECT_GT(results[i].texts[0].length(), prompts[i].length());
    }
}

// Stream generation test
TEST_F(GenerationTest, StreamGeneration) {
    std::vector<std::string> received_chunks;
    std::atomic<bool> generation_completed{false};
    
    auto callback = [&](const GenerationResult& chunk) {
        received_chunks.push_back(chunk.texts[0]);
        return true;  // Continue generation
    };
    
    config.stream = true;
    model->generate_stream("Test prompt", callback, config);
    
    EXPECT_FALSE(received_chunks.empty());
    for (const auto& chunk : received_chunks) {
        EXPECT_FALSE(chunk.empty());
    }
}

// Parameter effects test
TEST_F(GenerationTest, ParameterEffects) {
    const std::string prompt = "The quick brown fox";
    
    // Test temperature parameter
    config.temperature = 0.0f;
    auto result1 = model->generate(prompt, config);
    auto result2 = model->generate(prompt, config);
    EXPECT_EQ(result1.texts[0], result2.texts[0]);  // Should generate the same result
    
    // Test high temperature
    config.temperature = 1.0f;
    result1 = model->generate(prompt, config);
    result2 = model->generate(prompt, config);
    EXPECT_NE(result1.texts[0], result2.texts[0]);  // Should generate different results
    
    // Test top_p parameter
    config.temperature = 0.7f;
    config.top_p = 0.5f;
    auto result_low_p = model->generate(prompt, config);
    config.top_p = 0.9f;
    auto result_high_p = model->generate(prompt, config);
    EXPECT_NE(result_low_p.texts[0], result_high_p.texts[0]);
}

// Stop conditions test
TEST_F(GenerationTest, StopConditions) {
    config.stop_tokens = {"\n", "END"};
    auto result = model->generate("Generate a short sentence. END", config);
    
    EXPECT_FALSE(result.texts[0].empty());
    EXPECT_TRUE(result.stop_reasons);
    EXPECT_EQ(result.stop_reasons->size(), 1);
    EXPECT_TRUE(result.stop_reasons->front() == "stop_token" ||
                result.stop_reasons->front() == "max_tokens");
}

// Long text generation test
TEST_F(GenerationTest, LongTextGeneration) {
    config.max_tokens = 1000;
    auto result = model->generate("Write a long story about", config);
    
    EXPECT_FALSE(result.texts[0].empty());
    EXPECT_GT(result.texts[0].length(), 500);  // Assume a long enough text was generated
}

// Error recovery test
TEST_F(GenerationTest, ErrorRecovery) {
    // Simulate device error
    model->to_device("cpu");  // Ensure on CPU
    config.max_tokens = 1000000;  // Set a very large value
    
    EXPECT_THROW(model->generate("Test prompt", config), std::runtime_error);
    
    // Should be able to generate normally after recovery
    config.max_tokens = 20;
    EXPECT_NO_THROW(model->generate("Test prompt", config));
}

// Performance test
TEST_F(GenerationTest, PerformanceBenchmark) {
    const int num_iterations = 5;
    std::vector<double> generation_times;
    
    for (int i = 0; i < num_iterations; i++) {
        auto start_time = std::chrono::high_resolution_clock::now();
        auto result = model->generate("Test prompt", config);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();
        generation_times.push_back(duration);
    }
    
    // Calculate average generation time
    double avg_time = 0.0;
    for (double time : generation_times) {
        avg_time += time;
    }
    avg_time /= num_iterations;
    
    EXPECT_LT(avg_time, 1000.0);  // Assume generation time should be less than 1 second
}

// Concurrent generation test
TEST_F(GenerationTest, ConcurrentGeneration) {
    const int num_threads = 4;
    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};
    std::atomic<int> error_count{0};
    
    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back([this, &success_count, &error_count]() {
            try {
                auto result = model->generate("Test prompt", config);
                if (!result.texts.empty()) {
                    success_count++;
                }
            } catch (...) {
                error_count++;
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_EQ(success_count, num_threads);
    EXPECT_EQ(error_count, 0);
}

// Memory usage test
TEST_F(GenerationTest, MemoryUsage) {
    // Record initial memory usage
    size_t initial_memory = 0;  // TODO: Implement memory usage statistics
    
    // Generate a lot of text
    config.max_tokens = 500;
    auto result = model->generate("Generate a very long text", config);
    
    // Record peak memory usage
    size_t peak_memory = 0;  // TODO: Implement memory usage statistics
    
    // Memory usage after generation
    size_t final_memory = 0;  // TODO: Implement memory usage statistics
    
    EXPECT_GT(peak_memory, initial_memory);
    EXPECT_NEAR(final_memory, initial_memory, initial_memory * 0.1);  // Allow 10% difference
}

// Model switching test
TEST_F(GenerationTest, ModelSwitching) {
    auto result1 = model->generate("Test prompt", config);
    
    // Switch to another model
    model = std::make_shared<Model>("another_test_model");
    auto result2 = model->generate("Test prompt", config);
    
    EXPECT_NE(result1.texts[0], result2.texts[0]);
}

// Context length test
TEST_F(GenerationTest, ContextLength) {
    std::string long_prompt(model->max_sequence_length() * 2, 'a');  // Prompt longer than max length
    EXPECT_THROW(model->generate(long_prompt, config), std::invalid_argument);
    
    std::string valid_prompt(model->max_sequence_length() / 2, 'a');  // Valid length prompt
    EXPECT_NO_THROW(model->generate(valid_prompt, config));
} 