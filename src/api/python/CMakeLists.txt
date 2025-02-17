# Python bindings module
pybind11_add_module(_deeppowers
    deeppowers/bindings.cpp
)

# Link dependencies
target_link_libraries(_deeppowers
    PRIVATE
        deeppowers
)

# Set module properties
set_target_properties(_deeppowers PROPERTIES
    OUTPUT_NAME "_deeppowers"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/deeppowers"
)

# Install Python package
install(DIRECTORY deeppowers
    DESTINATION ${PYTHON_SITE_PACKAGES}
    FILES_MATCHING
        PATTERN "*.py"
        PATTERN "*.so"
        PATTERN "__pycache__" EXCLUDE
)

# Copy Python package to build directory
add_custom_command(TARGET _deeppowers POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/deeppowers
        ${CMAKE_BINARY_DIR}/python/deeppowers
)