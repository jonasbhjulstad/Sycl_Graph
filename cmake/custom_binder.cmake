function(add_custom_binder source_file)
    pybind11_add_module(${source_file} "${source_file}.cpp")
    target_link_libraries(${source_file} PUBLIC Static_RNG::Static_RNG TBB::tbb Eigen3::Eigen ortools::ortools)
    target_compile_options(${source_file} PRIVATE ${DEFAULT_WARNING_FLAGS})
    target_compile_options(${source_file} PUBLIC ${SYCL_CUSTOM_FLAGS} -fPIC)
    target_include_directories(${source_file} PUBLIC ${SYCL_GRAPH_INCLUDE_DIR})
    target_link_options(${source_file} PUBLIC ${SYCL_CUSTOM_FLAGS} -fPIC)
endfunction()




function(add_sycl_library source_file)
    add_library(${source_file} SHARED "${source_file}.cpp")
    # target_link_libraries(${source_file} PUBLIC Static_RNG::Static_RNG TBB::tbb Eigen3::Eigen)
    target_compile_options(${source_file} PUBLIC ${SYCL_CUSTOM_FLAGS})
    target_compile_options(${source_file} PUBLIC ${DEFAULT_WARNING_FLAGS})
    target_include_directories(${source_file} PUBLIC ${SYCL_GRAPH_INCLUDE_DIR})
    target_link_options(${source_file} PUBLIC ${SYCL_CUSTOM_FLAGS})
endfunction()


function(add_custom_library source_file)
    add_library(${source_file} SHARED "${source_file}.cpp")
    # target_link_libraries(${source_file} PUBLIC Static_RNG::Static_RNG TBB::tbb Eigen3::Eigen)
    # target_compile_options(${source_file} PUBLIC ${SYCL_CUSTOM_FLAGS})
    target_compile_options(${source_file} PUBLIC ${DEFAULT_WARNING_FLAGS})
    target_include_directories(${source_file} PUBLIC ${SYCL_GRAPH_INCLUDE_DIR})
    # target_link_options(${source_file} PUBLIC ${SYCL_CUSTOM_FLAGS})
endfunction()
