#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <random>
#include <memory>
#include <type_traits>

namespace deeppowers {
namespace common {

// StringUtils class
class StringUtils {
public:
    // Split string by delimiter
    static std::vector<std::string> split(const std::string& str, char delimiter);
    
    // Replace string by delimiter
    static std::string replace(const std::string& str, 
                             const std::string& from,
                             const std::string& to);
    
    // Convert string to lower case
    static std::string to_lower(const std::string& str);
    static std::string to_upper(const std::string& str);
    static std::string trim(const std::string& str);
    
    // Format string
    template<typename... Args>
    static std::string format(const std::string& format, Args... args);
};

// TimeUtils classUtils class
class TimeUtils {
public:
    // Get current timestamprent timestamp
    static int64_t current_time_ms();
    static int64_t current_time_us();
    
    // Format time
    static std::string format_time(const std::chrono::system_clock::time_point& time);
    static std::string format_duration(std::chrono::microseconds duration);
    
    // Timer class
    class Timer {
    public:
        Timer() : start_(std::chrono::high_resolution_clock::now()) {}
        
        void reset() {
            start_ = std::chrono::high_resolution_clock::now();
        }
        
        double elapsed_ms() const {
            auto now = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(now - start_).count();
        }
        
        double elapsed_us() const {
            auto now = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::micro>(now - start_).count();
        }
        
    private:
        std::chrono::high_resolution_clock::time_point start_;
    };
};

// MemoryUtils class
class MemoryUtils {
public:
    // Align memory
    static size_t align_up(size_t size, size_t alignment);
    
    // Format memory size
    static std::string format_size(size_t bytes);
    
    // Smart pointer tools
    template<typename T, typename... Args>
    static std::unique_ptr<T> make_unique_aligned(size_t alignment, Args&&... args);
    
    // Memory pool
    class MemoryPool {
    public:
        explicit MemoryPool(size_t block_size = 4096);
        ~MemoryPool();
        
        void* allocate(size_t size);
        void deallocate(void* ptr);
        
    private:
        struct Block;
        struct Chunk;
        
        size_t block_size_;
        Block* current_block_;
        Chunk* free_chunks_;
    };
};

// RandomUtils class
class RandomUtils {
public:
    // Get random number generator
    static std::mt19937& generator() {
        static thread_local std::mt19937 gen(std::random_device{}());
        return gen;
    }
    
    // Generate random integer
    static int random_int(int min, int max);
    
    // Generate random float
    static float random_float(float min, float max);
    
    // Generate random boolean
    static bool random_bool(float true_probability = 0.5f);
    
    // Generate random string
    static std::string random_string(size_t length);
    
    // Shuffle container
    template<typename Container>
    static void shuffle(Container& container);
};

// CUDAUtils class
class CUDAUtils {
public:
    // Get device properties
    static void get_device_properties(int device_id, cudaDeviceProp& prop);
    
    // Get device memory information
    static void get_memory_info(size_t& free, size_t& total);
    
    // Get SM count
    static int get_sm_count(int device_id);
    
    // Calculate grid and block dimensions
    static void get_grid_block_dim(int total_threads,
                                 dim3& grid_dim,
                                 dim3& block_dim,
                                 int max_threads_per_block = 256);
    
    // Check if CUDA is available
    static bool is_cuda_available();
    
    // Get best device
    static int get_best_device();
};

// MathUtils class
class MathUtils {
public:
    // Constants
    static constexpr float PI = 3.14159265358979323846f;
    static constexpr float E = 2.71828182845904523536f;
    
    // Basic math functions
    static float sigmoid(float x);
    static float tanh(float x);
    static float relu(float x);
    static float gelu(float x);
    
    // Vector operations
    template<typename T>
    static T dot_product(const std::vector<T>& a, const std::vector<T>& b);
    
    template<typename T>
    static std::vector<T> normalize(const std::vector<T>& vec);
    
    // Matrix operations
    template<typename T>
    static std::vector<std::vector<T>> matrix_multiply(
        const std::vector<std::vector<T>>& a,
        const std::vector<std::vector<T>>& b);
};

} // namespace common
} // namespace deeppowers 