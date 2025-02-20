# Create core library
add_library(deeppowers_core SHARED
    tokenizer/tokenizer.cpp
    model/model.cpp
    inference/inference.cpp
)

# Set include directories
target_include_directories(deeppowers_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Set library properties
set_target_properties(deeppowers_core PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
) 