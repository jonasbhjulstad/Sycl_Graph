function(add_custom_executable source_file)
    add_executable(${source_file} "${source_file}.cpp")
    target_link_libraries(${source_file} PUBLIC Sycl_Graph Static_RNG::Static_RNG oneDPL spdlog::spdlog)
    target_compile_options(${source_file} PUBLIC -fsycl -std=c++20)
    target_compile_options(${source_file} PUBLIC ${SYCL_COMPILE_OPTIONS})
    target_compile_options(${source_file} PUBLIC ${DEFAULT_WARNING_FLAGS})
    target_include_directories(${source_file} PUBLIC ${SYCL_INCLUDE_DIR})
#    if(${SYCL_GRAPH_USE_CUDA})
#        target_compile_options(${source_file} PRIVATE -fsycl ${SYCL_GRAPH_CUDA_FLAGS})
#        target_link_options(${source_file} PRIVATE -fsycl ${SYCL_GRAPH_CUDA_FLAGS})
#    endif()
    target_link_options(${source_file} PRIVATE ${DPCPP_FLAGS}-std=c++20)
endfunction()
