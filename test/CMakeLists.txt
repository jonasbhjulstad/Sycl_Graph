include(CTest)
enable_testing()
add_sycl_executable(random_connect_test)
target_link_libraries(random_connect_test PUBLIC Sycl_Graph)


add_test(NAME random_connect_test COMMAND random_connect_test)

add_sycl_executable(SBM_test)
target_link_libraries(SBM_test PUBLIC Sycl_Graph)

add_test(NAME SBM_test COMMAND SBM_test)
