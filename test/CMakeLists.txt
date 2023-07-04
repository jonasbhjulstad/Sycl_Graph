
include(CTest)

function(add_custom_test test_name)
    add_custom_executable(${test_name})
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

# add_subdirectory(Buffer)
add_subdirectory(Operations)
add_subdirectory(Epidemiological)
# add_subdirectory(Generation)
# add_subdirectory(Sycl)

add_subdirectory(Tuple)
