
include(CTest)

function(add_custom_test test_name)
    add_custom_executable(${test_name})
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

add_subdirectory(Buffer)

add_custom_executable(sycl_device_info)
add_custom_executable(sycl_copy_profiling)
add_custom_executable(sycl_kernel_emit)
add_custom_executable(tinymt_kernel)
add_custom_executable(invariant_graph)
add_custom_executable(eigen_sycl)
add_custom_executable(kernel_ordering)
add_custom_executable(property_extractor)
add_custom_executable(tuple_sort)
find_package(Eigen3 REQUIRED)
target_link_libraries(eigen_sycl PRIVATE Eigen3::Eigen)