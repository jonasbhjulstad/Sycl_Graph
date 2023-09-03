function(add_sycl_executable source_file)
    add_executable(${source_file} "${source_file}.cpp")
    target_link_libraries(${source_file} PRIVATE cppitertools TBB::tbb oneDPL)
    target_compile_options(${source_file} PRIVATE ${SYCL_CUSTOM_FLAGS})
    target_compile_options(${source_file} PRIVATE ${DEFAULT_WARNING_FLAGS})
    target_include_directories(${source_file} PRIVATE ${SYCL_GRAPH_INCLUDE_DIR})
    target_link_options(${source_file} PRIVATE ${SYCL_CUSTOM_FLAGS})
endfunction()

function(add_sycl_executable_nodep source_file)
    add_executable(${source_file} "${source_file}.cpp")
    target_compile_options(${source_file} PRIVATE ${SYCL_CUSTOM_FLAGS})
    target_compile_options(${source_file} PRIVATE ${DEFAULT_WARNING_FLAGS})
    target_include_directories(${source_file} PRIVATE ${SYCL_GRAPH_INCLUDE_DIR})
endfunction()

function(add_regression_executable source_file)
    add_executable(${source_file} "${source_file}.cpp")
    target_link_libraries(${source_file} PRIVATE Static_RNG::Static_RNG TBB::tbb Eigen3::Eigen ortools::ortools)
endfunction()
