# Add example executables
add_executable(basic_generation basic_generation.cpp)

# If other example files exist, add them
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/batch_generation.cpp)
    add_executable(batch_generation batch_generation.cpp)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/stream_generation.cpp)
    add_executable(stream_generation stream_generation.cpp)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/quantization_example.cpp)
    add_executable(quantization_example quantization_example.cpp)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/performance_benchmark.cpp)
    add_executable(performance_benchmark performance_benchmark.cpp)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/advanced_features.cpp)
    add_executable(advanced_features advanced_features.cpp)
endif()

# Get all example targets
file(GLOB EXAMPLE_SOURCES "*.cpp")
foreach(SOURCE ${EXAMPLE_SOURCES})
    get_filename_component(TARGET_NAME ${SOURCE} NAME_WE)
    if(TARGET ${TARGET_NAME})
        # Link dependencies
        target_link_libraries(${TARGET_NAME}
            PRIVATE
                deeppowers
        )
        
        # Set include directories
        target_include_directories(${TARGET_NAME}
            PRIVATE
                ${CMAKE_SOURCE_DIR}/src/api/cpp/include
        )
        
        # Copy example data
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/data)
            add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_directory
                    ${CMAKE_CURRENT_SOURCE_DIR}/data
                    $<TARGET_FILE_DIR:${TARGET_NAME}>/data
            )
        endif()
    endif()
endforeach() 