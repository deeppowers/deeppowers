# C++ API library
add_library(deeppowers SHARED
    src/deeppowers.cpp
)

# Set include directories
target_include_directories(deeppowers
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link dependencies
target_link_libraries(deeppowers
    PUBLIC
        deeppowers_core
)

# Set library properties
set_target_properties(deeppowers PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    EXPORT_NAME API
)

# Install targets
install(TARGETS deeppowers
    EXPORT deeppowers-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

# Install headers
install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

# Generate and install pkg-config file
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/deeppowers.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/deeppowers.pc
    @ONLY
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/deeppowers.pc
    DESTINATION lib/pkgconfig
) 