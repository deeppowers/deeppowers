#include "utils.hpp"
#include "error.hpp"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <sstream>
#include <iomanip>

namespace deeppowers {
namespace common {

// StringUtils implementation
std::vector<std::string> StringUtils::split(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

std::string StringUtils::replace(const std::string& str,
                               const std::string& from,
                               const std::string& to) {
    std::string result = str;
    size_t pos = 0;
    while ((pos = result.find(from, pos)) != std::string::npos) {
        result.replace(pos, from.length(), to);
        pos += to.length();
    }
    return result;
}

std::string StringUtils::to_lower(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

std::string StringUtils::to_upper(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(), ::toupper);
    return result;
}

std::string StringUtils::trim(const std::string& str) {
    auto start = std::find_if_not(str.begin(), str.end(), ::isspace);
    auto end = std::find_if_not(str.rbegin(), str.rend(), ::isspace).base();
    return (start < end ? std::string(start, end) : std::string());
}

// TimeUtils implementation
int64_t TimeUtils::current_time_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

int64_t TimeUtils::current_time_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string TimeUtils::format_time(const std::chrono::system_clock::time_point& time) {
    auto time_t = std::chrono::system_clock::to_time_t(time);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        time.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S")
       << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

std::string TimeUtils::format_duration(std::chrono::microseconds duration) {
    auto hours = std::chrono::duration_cast<std::chrono::hours>(duration);
    duration -= hours;
    auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration);
    duration -= minutes;
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
    duration -= seconds;
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
    
    std::stringstream ss;
    if (hours.count() > 0) {
        ss << hours.count() << "h ";
    }
    if (minutes.count() > 0) {
        ss << minutes.count() << "m ";
    }
    if (seconds.count() > 0) {
        ss << seconds.count() << "s ";
    }
    if (milliseconds.count() > 0) {
        ss << milliseconds.count() << "ms";
    }
    return ss.str();
}

// MemoryUtils implementation
size_t MemoryUtils::align_up(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

std::string MemoryUtils::format_size(size_t bytes) {
    static const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit = 0;
    double size = static_cast<double>(bytes);
    
    while (size >= 1024.0 && unit < 4) {
        size /= 1024.0;
        ++unit;
    }
    
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << size << " " << units[unit];
    return ss.str();
}

// RandomUtils implementation
int RandomUtils::random_int(int min, int max) {
    std::uniform_int_distribution<int> dist(min, max);
    return dist(generator());
}

float RandomUtils::random_float(float min, float max) {
    std::uniform_real_distribution<float> dist(min, max);
    return dist(generator());
}

bool RandomUtils::random_bool(float true_probability) {
    std::bernoulli_distribution dist(true_probability);
    return dist(generator());
}

std::string RandomUtils::random_string(size_t length) {
    static const char charset[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";
    
    std::string result;
    result.reserve(length);
    
    std::uniform_int_distribution<size_t> dist(0, sizeof(charset) - 2);
    for (size_t i = 0; i < length; ++i) {
        result += charset[dist(generator())];
    }
    
    return result;
}

// CUDAUtils implementation
void CUDAUtils::get_device_properties(int device_id, cudaDeviceProp& prop) {
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));
}

void CUDAUtils::get_memory_info(size_t& free, size_t& total) {
    CHECK_CUDA(cudaMemGetInfo(&free, &total));
}

int CUDAUtils::get_sm_count(int device_id) {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));
    return prop.multiProcessorCount;
}

void CUDAUtils::get_grid_block_dim(int total_threads,
                                 dim3& grid_dim,
                                 dim3& block_dim,
                                 int max_threads_per_block) {
    block_dim.x = std::min(total_threads, max_threads_per_block);
    grid_dim.x = (total_threads + block_dim.x - 1) / block_dim.x;
}

bool CUDAUtils::is_cuda_available() {
    int device_count;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    return error == cudaSuccess && device_count > 0;
}

int CUDAUtils::get_best_device() {
    int device_count;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    
    int best_device = 0;
    size_t max_memory = 0;
    
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, i));
        
        if (prop.totalGlobalMem > max_memory) {
            max_memory = prop.totalGlobalMem;
            best_device = i;
        }
    }
    
    return best_device;
}

// MathUtils implementation
float MathUtils::sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float MathUtils::tanh(float x) {
    return std::tanh(x);
}

float MathUtils::relu(float x) {
    return std::max(0.0f, x);
}

float MathUtils::gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coef = 0.044715f;
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + std::tanh(sqrt_2_over_pi * (x + coef * x3)));
}

template<typename T>
T MathUtils::dot_product(const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() != b.size()) {
        throw ArgumentException("Vectors must have the same size");
    }
    
    return std::inner_product(a.begin(), a.end(), b.begin(), T(0));
}

template<typename T>
std::vector<T> MathUtils::normalize(const std::vector<T>& vec) {
    T norm = std::sqrt(dot_product(vec, vec));
    if (norm == 0) return vec;
    
    std::vector<T> result(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        result[i] = vec[i] / norm;
    }
    return result;
}

template<typename T>
std::vector<std::vector<T>> MathUtils::matrix_multiply(
    const std::vector<std::vector<T>>& a,
    const std::vector<std::vector<T>>& b) {
    
    if (a.empty() || b.empty() || a[0].size() != b.size()) {
        throw ArgumentException("Invalid matrix dimensions");
    }
    
    size_t m = a.size();
    size_t n = b[0].size();
    size_t k = a[0].size();
    
    std::vector<std::vector<T>> result(m, std::vector<T>(n));
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            T sum = 0;
            for (size_t p = 0; p < k; ++p) {
                sum += a[i][p] * b[p][j];
            }
            result[i][j] = sum;
        }
    }
    
    return result;
}

// Explicitly instantiate templates
template float MathUtils::dot_product<float>(const std::vector<float>&, const std::vector<float>&);
template std::vector<float> MathUtils::normalize<float>(const std::vector<float>&);
template std::vector<std::vector<float>> MathUtils::matrix_multiply<float>(
    const std::vector<std::vector<float>>&,
    const std::vector<std::vector<float>>&);

std::string get_version() {
    return "0.1.0";
}

} // namespace common
} // namespace deeppowers 