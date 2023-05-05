
include(CTest)

function(add_custom_test test_name)
    add_custom_executable(${test_name})
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

add_subdirectory(Buffer)

add_custom_executable(sycl_device_info)
add_custom_executable(invariant_graph)
add_custom_executable(eigen_sycl)
add_custom_executable(property_extractor)
add_custom_executable(type_helpers_test)
add_custom_executable(graph_access)
add_executable(has_member has_member.cpp)
#set cxx standard to 20
target_compile_options(has_member PRIVATE -std=c++20)
find_package(Eigen3 REQUIRED)
target_link_libraries(eigen_sycl PRIVATE Eigen3::Eigen)