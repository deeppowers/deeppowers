# Add example executables
add_executable(basic_generation basic_generation.cpp)
add_executable(batch_generation batch_generation.cpp)
add_executable(stream_generation stream_generation.cpp)
add_executable(quantization_example quantization_example.cpp)
add_executable(performance_benchmark performance_benchmark.cpp)
add_executable(advanced_features advanced_features.cpp)

# Link dependencies
set(EXAMPLE_TARGETS
    basic_generation
    batch_generation
    stream_generation
    quantization_example
    performance_benchmark
    advanced_features
)

foreach(target ${EXAMPLE_TARGETS})
    target_link_libraries(${target}
        PRIVATE
            deeppowers
    )
    
    target_include_directories(${target}
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src/api/cpp/include
    )
    
    # Copy example data
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/data
            $<TARGET_FILE_DIR:${target}>/data
    )
endforeach() 