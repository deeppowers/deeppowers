#include <gtest/gtest.h>
#include <deeppowers.hpp>

using namespace deeppowers::api;

TEST(UtilsTest, VersionInfo) {
    // Test version string format
    std::string ver = version();
    EXPECT_FALSE(ver.empty());
    EXPECT_NE(ver.find_first_of("0123456789"), std::string::npos);
    EXPECT_NE(ver.find('.'), std::string::npos);
}

TEST(UtilsTest, CUDAInfo) {
    // Test CUDA version
    std::string cuda_ver = cuda_version();
    if (cuda_available()) {
        EXPECT_FALSE(cuda_ver.empty());
        EXPECT_NE(cuda_ver.find_first_of("0123456789"), std::string::npos);
        EXPECT_NE(cuda_ver.find('.'), std::string::npos);
    } else {
        EXPECT_TRUE(cuda_ver.empty());
    }
    
    // Test CUDA device count
    int device_count = cuda_device_count();
    if (cuda_available()) {
        EXPECT_GT(device_count, 0);
    } else {
        EXPECT_EQ(device_count, 0);
    }
}

TEST(UtilsTest, ModelManagement) {
    // Test model availability check
    EXPECT_NO_THROW(is_model_available("gpt2"));
    EXPECT_FALSE(is_model_available("non_existent_model"));
    
    // Test model listing
    auto models = list_available_models();
    EXPECT_FALSE(models.empty());
    
    // Test model loading
    EXPECT_THROW(load_model("non_existent_model"), std::runtime_error);
    if (!models.empty()) {
        EXPECT_NO_THROW(load_model(models[0]));
    }
} 