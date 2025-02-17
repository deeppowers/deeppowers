#include <gtest/gtest.h>
#include <deeppowers.hpp>

using namespace deeppowers::api;

TEST(GenerationConfigTest, DefaultValues) {
    GenerationConfig config;
    
    EXPECT_EQ(config.model_type, "gpt");
    EXPECT_EQ(config.max_tokens, 100);
    EXPECT_FLOAT_EQ(config.temperature, 0.7f);
    EXPECT_FLOAT_EQ(config.top_p, 1.0f);
    EXPECT_FLOAT_EQ(config.top_k, 0.0f);
    EXPECT_FALSE(config.stream);
    EXPECT_EQ(config.batch_size, 1);
    EXPECT_TRUE(config.stop_tokens.empty());
}

TEST(GenerationConfigTest, CustomValues) {
    GenerationConfig config;
    config.model_type = "llama";
    config.max_tokens = 200;
    config.temperature = 0.9f;
    config.top_p = 0.95f;
    config.top_k = 50.0f;
    config.stream = true;
    config.batch_size = 4;
    config.stop_tokens = {"\n", "###"};
    
    EXPECT_EQ(config.model_type, "llama");
    EXPECT_EQ(config.max_tokens, 200);
    EXPECT_FLOAT_EQ(config.temperature, 0.9f);
    EXPECT_FLOAT_EQ(config.top_p, 0.95f);
    EXPECT_FLOAT_EQ(config.top_k, 50.0f);
    EXPECT_TRUE(config.stream);
    EXPECT_EQ(config.batch_size, 4);
    EXPECT_EQ(config.stop_tokens.size(), 2);
    EXPECT_EQ(config.stop_tokens[0], "\n");
    EXPECT_EQ(config.stop_tokens[1], "###");
}

TEST(GenerationConfigTest, ValidationChecks) {
    GenerationConfig config;
    
    // Temperature validation
    config.temperature = -0.1f;
    EXPECT_THROW(config.validate(), std::invalid_argument);
    config.temperature = 1.5f;
    EXPECT_THROW(config.validate(), std::invalid_argument);
    config.temperature = 0.7f;
    
    // Top-p validation
    config.top_p = -0.1f;
    EXPECT_THROW(config.validate(), std::invalid_argument);
    config.top_p = 1.1f;
    EXPECT_THROW(config.validate(), std::invalid_argument);
    config.top_p = 0.95f;
    
    // Top-k validation
    config.top_k = -1.0f;
    EXPECT_THROW(config.validate(), std::invalid_argument);
    config.top_k = 50.0f;
    
    // Batch size validation
    config.batch_size = 0;
    EXPECT_THROW(config.validate(), std::invalid_argument);
    config.batch_size = -1;
    EXPECT_THROW(config.validate(), std::invalid_argument);
    config.batch_size = 1;
    
    // Valid configuration should not throw
    EXPECT_NO_THROW(config.validate());
} 