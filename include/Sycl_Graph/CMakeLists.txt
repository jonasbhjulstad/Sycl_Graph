add_library(Sycl_Graph STATIC path_config.cpp Math/math.cpp)
target_include_directories(Sycl_Graph PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>")
target_link_libraries(Sycl_Graph PRIVATE Tracy::TracyClient)
target_link_libraries(Sycl_Graph PUBLIC cppitertools::cppitertools oneDPL)
target_compile_options(Sycl_Graph PUBLIC ${DEFAULT_WARNING_FLAGS})
if(${SYCL_GRAPH_ENABLE_PRECOMPILED_HEADERS})
    #find all *.hpp and *.h files in include dirs
    set(SYCL_GRAPH_PCH_HEADERS)
    foreach(target Static_RNG Sycl_Graph)
        get_target_property(target_include_dirs ${target} INTERFACE_INCLUDE_DIRECTORIES)
        foreach(include_dir ${target_include_dirs})
            file(GLOB_RECURSE target_headers ${include_dir}/*.h ${include_dir}/*.hpp)
            list(APPEND SYCL_GRAPH_PCH_HEADERS ${target_headers})
        endforeach()
    endforeach()    

    #find all *.hpp and *.h files in Sycl_Graph_interface_include_dirs
    target_precompile_headers(Sycl_Graph PUBLIC ${SYCL_GRAPH_PCH_HEADERS})
endif()