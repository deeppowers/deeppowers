cmake_minimum_required(VERSION 3.15)
project(deeppowers_python)

# Find Python
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

# Add Python module
pybind11_add_module(_deeppowers_core
    deeppowers/bindings.cpp
)

# Link against core library
target_link_libraries(_deeppowers_core PRIVATE deeppowers_core)

# Include directories
target_include_directories(_deeppowers_core PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/include
    ${CMAKE_SOURCE_DIR}/src/common/include
)

# Set output directory
set_target_properties(_deeppowers_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/deeppowers
)

# Install rules
install(TARGETS _deeppowers_core DESTINATION deeppowers)
install(DIRECTORY deeppowers DESTINATION .
    FILES_MATCHING PATTERN "*.py"
)