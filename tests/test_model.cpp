#include <gtest/gtest.h>
#include <deeppowers.hpp>
#include <memory>

using namespace deeppowers::api;

class ModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a mock model for testing
        model = std::make_shared<Model>("test_model");
    }

    std::shared_ptr<Model> model;
};

TEST_F(ModelTest, ModelProperties) {
    EXPECT_EQ(model->model_path(), "test_model");
    EXPECT_EQ(model->model_type(), "gpt");
    EXPECT_GT(model->vocab_size(), 0);
    EXPECT_GT(model->max_sequence_length(), 0);
}

TEST_F(ModelTest, DeviceManagement) {
    EXPECT_EQ(model->device(), "cpu");
    
    // Test device movement
    if (cuda_available()) {
        model->to_device("cuda:0");
        EXPECT_EQ(model->device(), "cuda:0");
    }
    
    model->to_device("cpu");
    EXPECT_EQ(model->device(), "cpu");
    
    // Invalid device should throw
    EXPECT_THROW(model->to_device("invalid_device"), std::invalid_argument);
}

TEST_F(ModelTest, ConfigurationManagement) {
    // Test setting and getting configuration
    model->set_config("max_length", "1024");
    EXPECT_EQ(model->get_config("max_length"), "1024");
    
    // Test invalid configuration
    EXPECT_THROW(model->get_config("invalid_key"), std::out_of_range);
}

TEST_F(ModelTest, TextGeneration) {
    GenerationConfig config;
    config.max_tokens = 10;
    
    // Test basic generation
    auto result = model->generate("Hello, world!", config);
    EXPECT_FALSE(result.texts.empty());
    EXPECT_GT(result.generation_time, 0.0);
    
    // Test batch generation
    std::vector<std::string> prompts = {"Hello", "World"};
    auto batch_result = model->generate_batch(prompts, config);
    EXPECT_EQ(batch_result.texts.size(), 2);
    
    // Test streaming generation
    bool callback_called = false;
    auto stream_callback = [&callback_called](const std::string& text) {
        callback_called = true;
        return true;
    };
    
    model->generate_stream("Test prompt", stream_callback, config);
    EXPECT_TRUE(callback_called);
}

TEST_F(ModelTest, ErrorHandling) {
    GenerationConfig config;
    
    // Invalid prompt should throw
    EXPECT_THROW(model->generate("", config), std::invalid_argument);
    
    // Empty batch should throw
    std::vector<std::string> empty_prompts;
    EXPECT_THROW(model->generate_batch(empty_prompts, config), std::invalid_argument);
    
    // Invalid callback should throw
    EXPECT_THROW(model->generate_stream("Test", nullptr, config), std::invalid_argument);
} 