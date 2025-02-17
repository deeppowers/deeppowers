#include "quantization_manager.hpp"
#include <gtest/gtest.h>
#include <benchmark/benchmark.h>
#include <random>
#include <chrono>

namespace deeppowers {
namespace testing {

class QuantizationManagerPerfTest : public ::testing::Test {
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
    
    // Helper function to measure execution time
    template<typename F>
    double measure_time(F&& func) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
    
    std::unique_ptr<hal::Device> device_;
    std::unique_ptr<QuantizationManager> quant_manager_;
};

// Performance test for INT8 quantization
static void BM_Int8Quantization(benchmark::State& state) {
    // Setup
    auto device = std::make_unique<hal::CUDADevice>(0);
    auto quant_manager = std::make_unique<QuantizationManager>(device.get());
    
    // Create test tensor
    std::vector<int64_t> shape = {state.range(0), state.range(1), state.range(2)};
    auto tensor = std::make_unique<hal::Tensor>(shape, hal::DataType::FLOAT32, device.get());
    
    // Benchmark loop
    for (auto _ : state) {
        auto quantized = quant_manager->quantize_int8(tensor.get());
        benchmark::DoNotOptimize(quantized);
    }
    
    // Report metrics
    state.SetItemsProcessed(state.iterations() * tensor->size_in_bytes());
    state.SetBytesProcessed(state.iterations() * tensor->size_in_bytes());
}

// Performance test for INT4 quantization
static void BM_Int4Quantization(benchmark::State& state) {
    // Setup
    auto device = std::make_unique<hal::CUDADevice>(0);
    auto quant_manager = std::make_unique<QuantizationManager>(device.get());
    quant_manager->set_quantization_type(QuantizationType::INT4);
    
    // Create test tensor
    std::vector<int64_t> shape = {state.range(0), state.range(1), state.range(2)};
    auto tensor = std::make_unique<hal::Tensor>(shape, hal::DataType::FLOAT32, device.get());
    
    // Benchmark loop
    for (auto _ : state) {
        auto quantized = quant_manager->quantize_int4(tensor.get());
        benchmark::DoNotOptimize(quantized);
    }
    
    // Report metrics
    state.SetItemsProcessed(state.iterations() * tensor->size_in_bytes());
    state.SetBytesProcessed(state.iterations() * tensor->size_in_bytes());
}

// Performance test for per-channel quantization
static void BM_PerChannelQuantization(benchmark::State& state) {
    // Setup
    auto device = std::make_unique<hal::CUDADevice>(0);
    auto quant_manager = std::make_unique<QuantizationManager>(device.get());
    quant_manager->set_per_channel(true);
    
    // Create test tensor
    std::vector<int64_t> shape = {state.range(0), state.range(1), state.range(2)};
    auto tensor = std::make_unique<hal::Tensor>(shape, hal::DataType::FLOAT32, device.get());
    
    // Benchmark loop
    for (auto _ : state) {
        auto quantized = quant_manager->quantize_int8(tensor.get(), true);
        benchmark::DoNotOptimize(quantized);
    }
    
    // Report metrics
    state.SetItemsProcessed(state.iterations() * tensor->size_in_bytes());
    state.SetBytesProcessed(state.iterations() * tensor->size_in_bytes());
}

// Performance test for dequantization
static void BM_Dequantization(benchmark::State& state) {
    // Setup
    auto device = std::make_unique<hal::CUDADevice>(0);
    auto quant_manager = std::make_unique<QuantizationManager>(device.get());
    
    // Create and quantize test tensor
    std::vector<int64_t> shape = {state.range(0), state.range(1), state.range(2)};
    auto tensor = std::make_unique<hal::Tensor>(shape, hal::DataType::FLOAT32, device.get());
    auto quantized = quant_manager->quantize_int8(tensor.get());
    
    // Benchmark loop
    for (auto _ : state) {
        auto dequantized = quant_manager->dequantize(quantized.get());
        benchmark::DoNotOptimize(dequantized);
    }
    
    // Report metrics
    state.SetItemsProcessed(state.iterations() * tensor->size_in_bytes());
    state.SetBytesProcessed(state.iterations() * tensor->size_in_bytes());
}

// Performance test for calibration
static void BM_Calibration(benchmark::State& state) {
    // Setup
    auto device = std::make_unique<hal::CUDADevice>(0);
    auto quant_manager = std::make_unique<QuantizationManager>(device.get());
    
    // Create test tensor
    std::vector<int64_t> shape = {state.range(0), state.range(1), state.range(2)};
    auto tensor = std::make_unique<hal::Tensor>(shape, hal::DataType::FLOAT32, device.get());
    
    // Benchmark loop
    for (auto _ : state) {
        quant_manager->calibrate(tensor.get(), "test_tensor");
        benchmark::DoNotOptimize(quant_manager->is_calibrated());
    }
    
    // Report metrics
    state.SetItemsProcessed(state.iterations() * tensor->size_in_bytes());
    state.SetBytesProcessed(state.iterations() * tensor->size_in_bytes());
}

// Register benchmarks with different tensor sizes
BENCHMARK(BM_Int8Quantization)
    ->Args({32, 32, 32})
    ->Args({64, 64, 64})
    ->Args({128, 128, 128})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Int4Quantization)
    ->Args({32, 32, 32})
    ->Args({64, 64, 64})
    ->Args({128, 128, 128})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_PerChannelQuantization)
    ->Args({32, 32, 32})
    ->Args({64, 64, 64})
    ->Args({128, 128, 128})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Dequantization)
    ->Args({32, 32, 32})
    ->Args({64, 64, 64})
    ->Args({128, 128, 128})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Calibration)
    ->Args({32, 32, 32})
    ->Args({64, 64, 64})
    ->Args({128, 128, 128})
    ->Unit(benchmark::kMillisecond);

// Additional performance tests
TEST_F(QuantizationManagerPerfTest, QuantizationLatency) {
    // Create test tensor
    auto tensor = create_random_tensor({128, 128, 128});
    
    // Measure INT8 quantization latency
    double int8_latency = measure_time([&]() {
        auto quantized = quant_manager_->quantize_int8(tensor.get());
    });
    
    // Measure INT4 quantization latency
    quant_manager_->set_quantization_type(QuantizationType::INT4);
    double int4_latency = measure_time([&]() {
        auto quantized = quant_manager_->quantize_int4(tensor.get());
    });
    
    // Print results
    std::cout << "INT8 Quantization Latency: " << int8_latency << " ms\n";
    std::cout << "INT4 Quantization Latency: " << int4_latency << " ms\n";
    
    // Verify latencies are reasonable
    EXPECT_LT(int8_latency, 10.0);  // Less than 10ms
    EXPECT_LT(int4_latency, 10.0);  // Less than 10ms
}

TEST_F(QuantizationManagerPerfTest, ThroughputScaling) {
    std::vector<std::vector<int64_t>> shapes = {
        {32, 32, 32},
        {64, 64, 64},
        {128, 128, 128}
    };
    
    std::vector<double> throughputs;
    
    for (const auto& shape : shapes) {
        auto tensor = create_random_tensor(shape);
        size_t total_bytes = tensor->size_in_bytes();
        
        // Measure throughput
        double latency = measure_time([&]() {
            auto quantized = quant_manager_->quantize_int8(tensor.get());
        });
        
        double throughput = (total_bytes / (1024.0 * 1024.0)) / (latency / 1000.0);  // MB/s
        throughputs.push_back(throughput);
        
        std::cout << "Shape: " << shape[0] << "x" << shape[1] << "x" << shape[2]
                  << " Throughput: " << throughput << " MB/s\n";
    }
    
    // Verify throughput scaling
    for (size_t i = 1; i < throughputs.size(); ++i) {
        EXPECT_GT(throughputs[i], throughputs[i-1] * 0.7);  // At least 70% scaling
    }
}

} // namespace testing
} // namespace deeppowers 