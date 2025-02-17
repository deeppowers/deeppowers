#include "quantization_manager.hpp"
#include <gtest/gtest.h>
#include <nvml.h>
#include <thread>
#include <vector>

namespace deeppowers {
namespace testing {

class QuantizationManagerResourceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize NVML
        ASSERT_EQ(nvmlInit(), NVML_SUCCESS);
        
        // Create CUDA device
        device_ = std::make_unique<hal::CUDADevice>(0);
        
        // Create quantization manager
        quant_manager_ = std::make_unique<QuantizationManager>(device_.get());
        
        // Get NVML device handle
        ASSERT_EQ(nvmlDeviceGetHandleByIndex(0, &nvml_device_), NVML_SUCCESS);
    }
    
    void TearDown() override {
        // Shutdown NVML
        nvmlShutdown();
    }
    
    // Helper function to get GPU memory usage
    size_t get_gpu_memory_usage() {
        nvmlMemory_t memory;
        if (nvmlDeviceGetMemoryInfo(nvml_device_, &memory) != NVML_SUCCESS) {
            return 0;
        }
        return memory.used;
    }
    
    // Helper function to get GPU utilization
    float get_gpu_utilization() {
        nvmlUtilization_t utilization;
        if (nvmlDeviceGetUtilizationRates(nvml_device_, &utilization) != NVML_SUCCESS) {
            return 0.0f;
        }
        return utilization.gpu / 100.0f;
    }
    
    // Helper function to create tensor with specified size
    std::unique_ptr<hal::Tensor> create_tensor(size_t size_mb) {
        // Calculate shape for desired size
        size_t elements = (size_mb * 1024 * 1024) / sizeof(float);
        size_t dim = static_cast<size_t>(std::cbrt(elements));
        
        return std::make_unique<hal::Tensor>(
            std::vector<int64_t>{dim, dim, dim},
            hal::DataType::FLOAT32,
            device_.get());
    }
    
    std::unique_ptr<hal::Device> device_;
    std::unique_ptr<QuantizationManager> quant_manager_;
    nvmlDevice_t nvml_device_;
};

// Test memory usage for different tensor sizes
TEST_F(QuantizationManagerResourceTest, MemoryUsage) {
    std::vector<size_t> sizes_mb = {32, 64, 128, 256};
    std::vector<size_t> memory_usages;
    
    for (size_t size_mb : sizes_mb) {
        // Create test tensor
        auto tensor = create_tensor(size_mb);
        
        // Record initial memory usage
        size_t initial_usage = get_gpu_memory_usage();
        
        // Perform quantization
        auto quantized = quant_manager_->quantize_int8(tensor.get());
        
        // Record peak memory usage
        size_t peak_usage = get_gpu_memory_usage();
        memory_usages.push_back(peak_usage - initial_usage);
        
        std::cout << "Tensor size: " << size_mb << "MB, "
                  << "Memory overhead: " << (peak_usage - initial_usage) / (1024 * 1024) << "MB\n";
        
        // Verify memory overhead is reasonable
        EXPECT_LT(peak_usage - initial_usage, size_mb * 1024 * 1024 * 2);  // Less than 2x input size
    }
    
    // Verify memory scaling is roughly linear
    for (size_t i = 1; i < memory_usages.size(); ++i) {
        float scaling = static_cast<float>(memory_usages[i]) / memory_usages[i-1];
        EXPECT_NEAR(scaling, 2.0f, 0.3f);  // Should scale roughly 2x with tensor size
    }
}

// Test memory release after quantization
TEST_F(QuantizationManagerResourceTest, MemoryRelease) {
    // Create large test tensor
    auto tensor = create_tensor(256);  // 256MB
    
    // Record initial memory usage
    size_t initial_usage = get_gpu_memory_usage();
    
    // Perform multiple quantizations
    for (int i = 0; i < 5; ++i) {
        auto quantized = quant_manager_->quantize_int8(tensor.get());
    }
    
    // Force GPU synchronization
    device_->synchronize();
    
    // Record final memory usage
    size_t final_usage = get_gpu_memory_usage();
    
    // Verify memory is properly released
    EXPECT_NEAR(final_usage, initial_usage, 1024 * 1024);  // Within 1MB tolerance
}

// Test GPU utilization during quantization
TEST_F(QuantizationManagerResourceTest, GpuUtilization) {
    // Create test tensor
    auto tensor = create_tensor(128);  // 128MB
    
    // Measure GPU utilization during quantization
    std::vector<float> utilization_samples;
    std::atomic<bool> should_stop{false};
    
    // Start sampling thread
    std::thread sampling_thread([&]() {
        while (!should_stop) {
            utilization_samples.push_back(get_gpu_utilization());
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    });
    
    // Perform quantization
    auto quantized = quant_manager_->quantize_int8(tensor.get());
    
    // Stop sampling
    should_stop = true;
    sampling_thread.join();
    
    // Calculate average utilization
    float avg_utilization = 0.0f;
    for (float util : utilization_samples) {
        avg_utilization += util;
    }
    avg_utilization /= utilization_samples.size();
    
    std::cout << "Average GPU utilization: " << avg_utilization * 100 << "%\n";
    
    // Verify GPU is reasonably utilized
    EXPECT_GT(avg_utilization, 0.3f);  // At least 30% utilization
}

// Test concurrent quantization
TEST_F(QuantizationManagerResourceTest, ConcurrentQuantization) {
    const int num_threads = 4;
    const int iterations = 10;
    
    // Create test tensors
    std::vector<std::unique_ptr<hal::Tensor>> tensors;
    for (int i = 0; i < num_threads; ++i) {
        tensors.push_back(create_tensor(64));  // 64MB each
    }
    
    // Record initial memory usage
    size_t initial_usage = get_gpu_memory_usage();
    
    // Launch concurrent quantization threads
    std::vector<std::thread> threads;
    std::atomic<int> error_count{0};
    
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            try {
                for (int j = 0; j < iterations; ++j) {
                    auto quantized = quant_manager_->quantize_int8(tensors[i].get());
                    device_->synchronize();
                }
            } catch (...) {
                error_count++;
            }
        });
    }
    
    // Wait for completion
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Verify no errors occurred
    EXPECT_EQ(error_count, 0);
    
    // Record peak memory usage
    size_t peak_usage = get_gpu_memory_usage();
    
    // Verify memory usage is reasonable
    size_t expected_usage = 64 * 1024 * 1024 * num_threads * 2;  // 2x buffer for each tensor
    EXPECT_LT(peak_usage - initial_usage, expected_usage);
}

// Test memory pressure handling
TEST_F(QuantizationManagerResourceTest, MemoryPressure) {
    // Create large tensors to consume memory
    std::vector<std::unique_ptr<hal::Tensor>> background_tensors;
    size_t available_memory = device_->available_memory();
    size_t tensor_size = available_memory / 4;  // Use 1/4 of available memory per tensor
    
    // Allocate background tensors until 75% memory is used
    while (get_gpu_memory_usage() < available_memory * 0.75) {
        try {
            background_tensors.push_back(create_tensor(tensor_size / (1024 * 1024)));
        } catch (...) {
            break;
        }
    }
    
    // Try quantization under memory pressure
    auto test_tensor = create_tensor(64);  // 64MB
    auto quantized = quant_manager_->quantize_int8(test_tensor.get());
    
    // Verify quantization succeeded
    ASSERT_NE(quantized, nullptr);
    
    // Release background tensors
    background_tensors.clear();
    
    // Verify memory is released
    device_->synchronize();
    EXPECT_LT(get_gpu_memory_usage(), available_memory * 0.5);
}

} // namespace testing
} // namespace deeppowers 