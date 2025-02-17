#include <deeppowers.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <numeric>
#include <algorithm>
#include <iomanip>

using namespace deeppowers::api;
using namespace std::chrono;

// Performance test configuration
struct BenchmarkConfig {
    size_t num_iterations = 10;
    size_t num_warmup_iterations = 3;
    std::vector<size_t> batch_sizes = {1, 2, 4, 8, 16};
    std::vector<size_t> sequence_lengths = {32, 64, 128, 256};
    bool test_cuda = true;
    bool test_quantization = true;
};

// Performance test results
struct BenchmarkResult {
    double avg_latency_ms = 0.0;
    double min_latency_ms = std::numeric_limits<double>::max();
    double max_latency_ms = 0.0;
    double p90_latency_ms = 0.0;
    double p95_latency_ms = 0.0;
    double p99_latency_ms = 0.0;
    double throughput = 0.0;
    double memory_usage_mb = 0.0;
};

// Helper function: Generate random prompt text of specified length
std::string generate_random_prompt(size_t length) {
    static const std::string chars = 
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ";
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, chars.size() - 1);
    
    std::string prompt;
    prompt.reserve(length);
    for (size_t i = 0; i < length; i++) {
        prompt += chars[dis(gen)];
    }
    return prompt;
}

// Run single benchmark test
BenchmarkResult run_benchmark(Model* model,
                            const GenerationConfig& config,
                            size_t batch_size,
                            size_t sequence_length,
                            const BenchmarkConfig& bench_config) {
    std::vector<double> latencies;
    latencies.reserve(bench_config.num_iterations);
    
    // Prepare test data
    std::vector<std::string> prompts;
    for (size_t i = 0; i < batch_size; i++) {
        prompts.push_back(generate_random_prompt(sequence_length));
    }
    
    // Warm-up run
    for (size_t i = 0; i < bench_config.num_warmup_iterations; i++) {
        if (batch_size == 1) {
            model->generate(prompts[0], config);
        } else {
            model->generate_batch(prompts, config);
        }
    }
    
    // Main benchmark test
    size_t total_tokens = 0;
    for (size_t i = 0; i < bench_config.num_iterations; i++) {
        auto start_time = high_resolution_clock::now();
        
        if (batch_size == 1) {
            auto result = model->generate(prompts[0], config);
            total_tokens += result.texts[0].length();
        } else {
            auto results = model->generate_batch(prompts, config);
            for (const auto& result : results) {
                total_tokens += result.texts[0].length();
            }
        }
        
        auto end_time = high_resolution_clock::now();
        auto latency = duration_cast<microseconds>(end_time - start_time).count() / 1000.0;
        latencies.push_back(latency);
    }
    
    // Calculate statistics
    BenchmarkResult result;
    
    // Calculate average latency
    result.avg_latency_ms = std::accumulate(latencies.begin(), latencies.end(), 0.0) / 
                           latencies.size();
    
    // Calculate minimum/maximum latency
    result.min_latency_ms = *std::min_element(latencies.begin(), latencies.end());
    result.max_latency_ms = *std::max_element(latencies.begin(), latencies.end());
    
    // Calculate percentile latency
    std::sort(latencies.begin(), latencies.end());
    size_t p90_idx = static_cast<size_t>(latencies.size() * 0.9);
    size_t p95_idx = static_cast<size_t>(latencies.size() * 0.95);
    size_t p99_idx = static_cast<size_t>(latencies.size() * 0.99);
    
    result.p90_latency_ms = latencies[p90_idx];
    result.p95_latency_ms = latencies[p95_idx];
    result.p99_latency_ms = latencies[p99_idx];
    
    // Calculate throughput (tokens/sec)
    double total_time = std::accumulate(latencies.begin(), latencies.end(), 0.0) / 1000.0;
    result.throughput = total_tokens / total_time;
    
    // Get memory usage
    result.memory_usage_mb = model->get_memory_usage() / (1024.0 * 1024.0);
    
    return result;
}

// Print results table
void print_results(const std::string& test_name,
                  size_t batch_size,
                  size_t sequence_length,
                  const BenchmarkResult& result) {
    std::cout << "\n" << test_name << std::endl;
    std::cout << "Batch size: " << batch_size 
              << ", Sequence length: " << sequence_length << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Average latency: " << result.avg_latency_ms << " ms" << std::endl;
    std::cout << "Min latency: " << result.min_latency_ms << " ms" << std::endl;
    std::cout << "Max latency: " << result.max_latency_ms << " ms" << std::endl;
    std::cout << "P90 latency: " << result.p90_latency_ms << " ms" << std::endl;
    std::cout << "P95 latency: " << result.p95_latency_ms << " ms" << std::endl;
    std::cout << "P99 latency: " << result.p99_latency_ms << " ms" << std::endl;
    std::cout << "Throughput: " << result.throughput << " tokens/sec" << std::endl;
    std::cout << "Memory usage: " << result.memory_usage_mb << " MB" << std::endl;
}

int main() {
    try {
        // Initialize model
        std::cout << "Loading model..." << std::endl;
        auto model = load_model("gpt2");
        
        // Set benchmark test configuration
        BenchmarkConfig bench_config;
        
        // Basic generation configuration
        GenerationConfig gen_config;
        gen_config.max_tokens = 50;
        gen_config.temperature = 0.0f;  // Use deterministic generation for consistent results
        
        // Run CPU benchmark test
        std::cout << "\nRunning CPU benchmarks..." << std::endl;
        model->to_device("cpu");
        
        for (size_t batch_size : bench_config.batch_sizes) {
            for (size_t seq_len : bench_config.sequence_lengths) {
                gen_config.batch_size = batch_size;
                auto result = run_benchmark(model.get(), gen_config, batch_size, 
                                         seq_len, bench_config);
                print_results("CPU", batch_size, seq_len, result);
            }
        }
        
        // Run CUDA benchmark test
        if (bench_config.test_cuda && cuda_available()) {
            std::cout << "\nRunning CUDA benchmarks..." << std::endl;
            model->to_device("cuda");
            
            for (size_t batch_size : bench_config.batch_sizes) {
                for (size_t seq_len : bench_config.sequence_lengths) {
                    gen_config.batch_size = batch_size;
                    auto result = run_benchmark(model.get(), gen_config, batch_size, 
                                             seq_len, bench_config);
                    print_results("CUDA", batch_size, seq_len, result);
                }
            }
        }
        
        // Run quantized model benchmark test
        if (bench_config.test_quantization) {
            std::cout << "\nRunning quantized model benchmarks..." << std::endl;
            
            // Configure INT8 quantization
            QuantizationConfig quant_config;
            quant_config.type = QuantizationType::INT8;
            quant_config.method = QuantizationMethod::POST_TRAINING;
            
            model->set_quantization_config(quant_config);
            model->quantize({"Sample calibration data"});  // Use sample calibration data
            
            for (size_t batch_size : bench_config.batch_sizes) {
                for (size_t seq_len : bench_config.sequence_lengths) {
                    gen_config.batch_size = batch_size;
                    auto result = run_benchmark(model.get(), gen_config, batch_size, 
                                             seq_len, bench_config);
                    print_results("INT8 Quantized", batch_size, seq_len, result);
                }
            }
        }
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
} 