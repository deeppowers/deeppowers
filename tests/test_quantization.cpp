#include <gtest/gtest.h>
#include <deeppowers.hpp>
#include <memory>
#include <random>

using namespace deeppowers::api;

class QuantizationTest : public ::testing::Test {
protected:
    void SetUp() override {
        model = std::make_shared<Model>("test_model");
        config = GenerationConfig();
    }

    std::shared_ptr<Model> model;
    GenerationConfig config;
    
    // Helper function: Generate random calibration data
    std::vector<std::string> generate_calibration_data(size_t num_samples) {
        std::vector<std::string> data;
        std::vector<std::string> templates = {
            "The quick brown fox",
            "Hello world",
            "This is a test",
            "Machine learning is"
        };
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, templates.size() - 1);
        
        for (size_t i = 0; i < num_samples; i++) {
            data.push_back(templates[dis(gen)]);
        }
        
        return data;
    }
};

// INT8 quantization test
TEST_F(QuantizationTest, Int8Quantization) {
    // Prepare calibration data
    auto calibration_data = generate_calibration_data(100);
    
    // Execute INT8 quantization
    model->quantize(QuantizationType::INT8, calibration_data);
    
    // Verify performance after quantization
    auto result = model->generate("Test prompt", config);
    EXPECT_FALSE(result.texts.empty());
    
    // Verify model size reduction
    EXPECT_LT(model->get_model_size(), model->get_original_model_size());
}

// INT4 quantization test
TEST_F(QuantizationTest, Int4Quantization) {
    auto calibration_data = generate_calibration_data(100);
    
    model->quantize(QuantizationType::INT4, calibration_data);
    
    auto result = model->generate("Test prompt", config);
    EXPECT_FALSE(result.texts.empty());
    
    // INT4 should save more space than INT8
    EXPECT_LT(model->get_model_size(), model->get_original_model_size() * 0.3);
}

// Quantization accuracy test
TEST_F(QuantizationTest, QuantizationAccuracy) {
    // Generate reference results
    const std::string prompt = "Test prompt for accuracy";
    auto reference_result = model->generate(prompt, config);
    
    // INT8 quantization
    model->quantize(QuantizationType::INT8, generate_calibration_data(100));
    auto int8_result = model->generate(prompt, config);
    
    // Calculate similarity (using simple character matching, should use a more complex metric in practice)
    float int8_similarity = 0.0f;
    // TODO: Implement similarity calculation
    
    EXPECT_GT(int8_similarity, 0.8f);  // Assume 80% similarity is required
    
    // INT4 quantization
    model->reset();  // Reset to original model
    model->quantize(QuantizationType::INT4, generate_calibration_data(100));
    auto int4_result = model->generate(prompt, config);
    
    float int4_similarity = 0.0f;
    // TODO: Implement similarity calculation
    
    EXPECT_GT(int4_similarity, 0.7f);  // INT4 can accept slightly lower similarity
}

// Quantization performance test
TEST_F(GenerationTest, QuantizedPerformance) {
    const int num_iterations = 5;
    const std::string prompt = "Test prompt for performance";
    
    // Test original model performance
    std::vector<double> original_times;
    for (int i = 0; i < num_iterations; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        model->generate(prompt, config);
        auto end = std::chrono::high_resolution_clock::now();
        original_times.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(
            end - start).count());
    }
    
    // Performance after INT8 quantization
    model->quantize(QuantizationType::INT8, generate_calibration_data(100));
    std::vector<double> int8_times;
    for (int i = 0; i < num_iterations; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        model->generate(prompt, config);
        auto end = std::chrono::high_resolution_clock::now();
        int8_times.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(
            end - start).count());
    }
    
    // Calculate average time
    double avg_original = std::accumulate(original_times.begin(), 
                                        original_times.end(), 0.0) / num_iterations;
    double avg_int8 = std::accumulate(int8_times.begin(), 
                                    int8_times.end(), 0.0) / num_iterations;
    
    // Quantization should be faster
    EXPECT_LT(avg_int8, avg_original);
}

// Quantization configuration test
TEST_F(QuantizationTest, QuantizationConfig) {
    QuantizationConfig quant_config;
    quant_config.type = QuantizationType::INT8;
    quant_config.method = QuantizationMethod::POST_TRAINING;
    quant_config.per_channel = true;
    quant_config.symmetric = true;
    
    model->set_quantization_config(quant_config);
    model->quantize(generate_calibration_data(100));
    
    // Verify configuration is correctly applied
    auto applied_config = model->get_quantization_config();
    EXPECT_EQ(applied_config.type, quant_config.type);
    EXPECT_EQ(applied_config.method, quant_config.method);
    EXPECT_EQ(applied_config.per_channel, quant_config.per_channel);
    EXPECT_EQ(applied_config.symmetric, quant_config.symmetric);
}

// Quantization persistence test
TEST_F(QuantizationTest, QuantizationPersistence) {
    // Quantize model
    model->quantize(QuantizationType::INT8, generate_calibration_data(100));
    
    // Save quantized model
    const std::string save_path = "quantized_model.bin";
    model->save(save_path);
    
    // Load quantized model
    auto loaded_model = std::make_shared<Model>(save_path);
    
    // Verify loaded model is in quantized state
    EXPECT_TRUE(loaded_model->is_quantized());
    EXPECT_EQ(loaded_model->get_quantization_config().type, QuantizationType::INT8);
    
    // Verify generation results
    auto result = loaded_model->generate("Test prompt", config);
    EXPECT_FALSE(result.texts.empty());
}

// Quantization error handling test
TEST_F(QuantizationTest, QuantizationErrorHandling) {
    // Test invalid quantization type
    EXPECT_THROW(model->quantize(static_cast<QuantizationType>(999), 
                                generate_calibration_data(100)),
                 std::invalid_argument);
    
    // Test empty calibration data
    std::vector<std::string> empty_data;
    EXPECT_THROW(model->quantize(QuantizationType::INT8, empty_data),
                 std::invalid_argument);
    
    // Test invalid calibration data
    std::vector<std::string> invalid_data(100, "");  // Empty strings
    EXPECT_THROW(model->quantize(QuantizationType::INT8, invalid_data),
                 std::invalid_argument);
}

// Quantization recovery test
TEST_F(QuantizationTest, QuantizationRecovery) {
    // Save original generation results
    const std::string prompt = "Test prompt for recovery";
    auto original_result = model->generate(prompt, config);
    
    // Quantize model
    model->quantize(QuantizationType::INT8, generate_calibration_data(100));
    
    // Recover to FP32
    model->dequantize();
    
    // Verify recovered results
    auto recovered_result = model->generate(prompt, config);
    EXPECT_EQ(original_result.texts[0], recovered_result.texts[0]);
}

// Mixed precision quantization test
TEST_F(QuantizationTest, MixedPrecisionQuantization) {
    QuantizationConfig quant_config;
    quant_config.type = QuantizationType::MIXED;
    quant_config.method = QuantizationMethod::DYNAMIC;
    
    // Set different layer quantization precisions
    std::unordered_map<std::string, QuantizationType> layer_precisions;
    layer_precisions["attention"] = QuantizationType::INT8;
    layer_precisions["ffn"] = QuantizationType::INT4;
    layer_precisions["embedding"] = QuantizationType::NONE;
    
    quant_config.layer_precisions = layer_precisions;
    
    model->set_quantization_config(quant_config);
    model->quantize(generate_calibration_data(100));
    
    // Verify different layer quantization precisions
    auto layer_info = model->get_layer_quantization_info();
    EXPECT_EQ(layer_info["attention"].type, QuantizationType::INT8);
    EXPECT_EQ(layer_info["ffn"].type, QuantizationType::INT4);
    EXPECT_EQ(layer_info["embedding"].type, QuantizationType::NONE);
} 