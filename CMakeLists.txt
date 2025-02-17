cmake_minimum_required(VERSION 3.15)
project(deeppowers VERSION 0.1.0 LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86")

# Find dependencies
find_package(CUDA REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(pybind11 REQUIRED)
find_package(nlohmann_json REQUIRED)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Add subdirectories
add_subdirectory(src/core)
add_subdirectory(src/api/cpp)
add_subdirectory(src/api/python)

# Install targets
install(EXPORT deeppowers-targets
        FILE deeppowers-targets.cmake
        NAMESPACE deeppowers::
        DESTINATION lib/cmake/deeppowers)

# Generate and install config file
include(CMakePackageConfigHelpers)
configure_package_config_file(
    cmake/deeppowers-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/deeppowers-config.cmake
    INSTALL_DESTINATION lib/cmake/deeppowers
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/deeppowers-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/deeppowers-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/deeppowers-config-version.cmake
    DESTINATION lib/cmake/deeppowers
)

# Add tests
enable_testing()
add_subdirectory(tests)