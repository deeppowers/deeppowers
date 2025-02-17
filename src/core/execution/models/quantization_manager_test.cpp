#include "quantization_manager.hpp"
#include <gtest/gtest.h>
#include <random>

namespace deeppowers {
namespace testing {

class QuantizationManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create CUDA device
        device_ = std::make_unique<hal::CUDADevice>(0);
        
        // Create quantization manager
        quant_manager_ = std::make_unique<QuantizationManager>(device_.get());
        
        // Set default configuration
        quant_manager_->set_quantization_type(QuantizationType::INT8);
        quant_manager_->set_quantization_method(QuantizationMethod::POST_TRAINING);
        quant_manager_->set_per_channel(false);
        quant_manager_->set_symmetric(true);
    }
    
    // Helper function to create random tensor
    std::unique_ptr<hal::Tensor> create_random_tensor(
        const std::vector<int64_t>& shape,
        float min_val = -1.0f,
        float max_val = 1.0f) {
        
        // Calculate total size
        size_t num_elements = 1;
        for (int64_t dim : shape) {
            num_elements *= dim;
        }
        
        // Generate random data
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(min_val, max_val);
        
        std::vector<float> host_data(num_elements);
        for (size_t i = 0; i < num_elements; ++i) {
            host_data[i] = dis(gen);
        }
        
        // Create and initialize tensor
        auto tensor = std::make_unique<hal::Tensor>(
            shape, hal::DataType::FLOAT32, device_.get());
        tensor->copy_from_host(host_data.data());
        
        return tensor;
    }
    
    // Helper function to compare tensors
    void compare_tensors(
        const hal::Tensor* a,
        const hal::Tensor* b,
        float tolerance = 1e-6f) {
        
        ASSERT_EQ(a->shape(), b->shape());
        ASSERT_EQ(a->dtype(), b->dtype());
        
        size_t num_elements = 1;
        for (int64_t dim : a->shape()) {
            num_elements *= dim;
        }
        
        std::vector<float> a_data(num_elements);
        std::vector<float> b_data(num_elements);
        
        a->copy_to_host(a_data.data());
        b->copy_to_host(b_data.data());
        
        for (size_t i = 0; i < num_elements; ++i) {
            EXPECT_NEAR(a_data[i], b_data[i], tolerance);
        }
    }
    
    std::unique_ptr<hal::Device> device_;
    std::unique_ptr<QuantizationManager> quant_manager_;
};

TEST_F(QuantizationManagerTest, Int8Quantization) {
    // Create test tensor
    auto tensor = create_random_tensor({2, 3, 4});
    
    // Perform INT8 quantization
    auto quantized = quant_manager_->quantize_int8(tensor.get());
    ASSERT_NE(quantized, nullptr);
    EXPECT_EQ(quantized->dtype(), hal::DataType::INT8);
    EXPECT_EQ(quantized->shape(), tensor->shape());
    
    // Perform dequantization
    auto dequantized = quant_manager_->dequantize(quantized.get());
    ASSERT_NE(dequantized, nullptr);
    EXPECT_EQ(dequantized->dtype(), hal::DataType::FLOAT32);
    
    // Compare original and reconstructed data
    compare_tensors(tensor.get(), dequantized.get(), 0.1f);  // Larger tolerance for INT8
}

TEST_F(QuantizationManagerTest, Int4Quantization) {
    // Create test tensor
    auto tensor = create_random_tensor({2, 3, 4});
    
    // Set INT4 quantization
    quant_manager_->set_quantization_type(QuantizationType::INT4);
    
    // Perform INT4 quantization
    auto quantized = quant_manager_->quantize_int4(tensor.get());
    ASSERT_NE(quantized, nullptr);
    EXPECT_EQ(quantized->dtype(), hal::DataType::INT8);  // Packed INT4
    
    // Check packed shape
    auto packed_shape = tensor->shape();
    packed_shape.back() = (packed_shape.back() + 1) / 2;
    EXPECT_EQ(quantized->shape(), packed_shape);
    
    // Perform dequantization
    auto dequantized = quant_manager_->dequantize(quantized.get());
    ASSERT_NE(dequantized, nullptr);
    EXPECT_EQ(dequantized->dtype(), hal::DataType::FLOAT32);
    
    // Compare original and reconstructed data
    compare_tensors(tensor.get(), dequantized.get(), 0.2f);  // Larger tolerance for INT4
}

TEST_F(QuantizationManagerTest, PerChannelQuantization) {
    // Create test tensor
    auto tensor = create_random_tensor({2, 3, 4});
    
    // Enable per-channel quantization
    quant_manager_->set_per_channel(true);
    
    // Perform quantization
    auto quantized = quant_manager_->quantize_int8(tensor.get(), true);
    ASSERT_NE(quantized, nullptr);
    
    // Check scales and zero points
    EXPECT_EQ(quantized->scales().size(), tensor->shape().back());
    EXPECT_EQ(quantized->zero_points().size(), tensor->shape().back());
    
    // Perform dequantization
    auto dequantized = quant_manager_->dequantize(quantized.get());
    ASSERT_NE(dequantized, nullptr);
    
    // Compare original and reconstructed data
    compare_tensors(tensor.get(), dequantized.get(), 0.1f);
}

TEST_F(QuantizationManagerTest, Calibration) {
    // Create test tensors
    auto tensor1 = create_random_tensor({2, 3, 4}, -2.0f, 2.0f);
    auto tensor2 = create_random_tensor({2, 3, 4}, -1.5f, 1.5f);
    
    // Perform calibration
    quant_manager_->calibrate(tensor1.get(), "test_tensor");
    quant_manager_->calibrate(tensor2.get(), "test_tensor");
    quant_manager_->finalize_calibration();
    
    EXPECT_TRUE(quant_manager_->is_calibrated());
    
    // Perform quantization using calibrated parameters
    auto quantized = quant_manager_->quantize_int8(tensor1.get());
    ASSERT_NE(quantized, nullptr);
    
    // Perform dequantization
    auto dequantized = quant_manager_->dequantize(quantized.get());
    ASSERT_NE(dequantized, nullptr);
    
    // Compare original and reconstructed data
    compare_tensors(tensor1.get(), dequantized.get(), 0.1f);
}

TEST_F(QuantizationManagerTest, DynamicQuantization) {
    // Set dynamic quantization
    quant_manager_->set_quantization_method(QuantizationMethod::DYNAMIC);
    
    // Create test tensor
    auto tensor = create_random_tensor({2, 3, 4});
    
    // Perform dynamic quantization
    auto quantized = quant_manager_->maybe_quantize(tensor.get(), "test_tensor");
    ASSERT_NE(quantized, nullptr);
    
    // Perform dequantization
    auto dequantized = quant_manager_->maybe_dequantize(quantized.get(), "test_tensor");
    ASSERT_NE(dequantized, nullptr);
    
    // Compare original and reconstructed data
    compare_tensors(tensor.get(), dequantized.get(), 0.1f);
}

TEST_F(QuantizationManagerTest, AsymmetricQuantization) {
    // Enable asymmetric quantization
    quant_manager_->set_symmetric(false);
    
    // Create test tensor with non-zero mean
    auto tensor = create_random_tensor({2, 3, 4}, 0.5f, 1.5f);
    
    // Perform quantization
    auto quantized = quant_manager_->quantize_int8(tensor.get());
    ASSERT_NE(quantized, nullptr);
    
    // Check zero points are non-zero
    const auto& zero_points = quantized->zero_points();
    bool has_nonzero = false;
    for (int8_t zp : zero_points) {
        if (zp != 0) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
    
    // Perform dequantization
    auto dequantized = quant_manager_->dequantize(quantized.get());
    ASSERT_NE(dequantized, nullptr);
    
    // Compare original and reconstructed data
    compare_tensors(tensor.get(), dequantized.get(), 0.1f);
}

} // namespace testing
} // namespace deeppowers 