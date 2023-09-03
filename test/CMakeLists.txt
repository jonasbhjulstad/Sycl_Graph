include(CTest)
enable_testing()
add_sycl_executable(random_connect_test)
target_link_libraries(random_connect_test PUBLIC Generation Metrics)


add_test(NAME random_connect_test COMMAND random_connect_test)
