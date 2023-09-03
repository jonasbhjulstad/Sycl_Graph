add_sycl_library(Complete_Graph)
target_link_libraries(Complete_Graph PRIVATE TBB::tbb cppitertools)
add_sycl_library(Random_Connect)
target_link_libraries(Random_Connect PUBLIC oneDPL)
target_link_libraries(Random_Connect PUBLIC Math)

add_library(Generation INTERFACE)
target_link_libraries(Generation INTERFACE Complete_Graph Random_Connect)
