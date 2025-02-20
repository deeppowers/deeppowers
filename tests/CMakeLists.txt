# Find GTest package
find_package(GTest REQUIRED)

# Add test executable
add_executable(deeppowers_tests
    test_utils.cpp
)

# Link dependencies
target_link_libraries(deeppowers_tests
    PRIVATE
        deeppowers
        GTest::GTest
        GTest::Main
)

# Set include directories
target_include_directories(deeppowers_tests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/api/cpp/include
)

# Add to CTest
include(GoogleTest)
gtest_discover_tests(deeppowers_tests)

# Copy test data
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR}) 