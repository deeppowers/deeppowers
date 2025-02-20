# Create common library
add_library(deeppowers_common STATIC
    utils.cpp
)

# Set include directories
target_include_directories(deeppowers_common
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
) 