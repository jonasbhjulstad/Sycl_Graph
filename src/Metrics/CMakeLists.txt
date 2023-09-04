add_sycl_library(Degrees)
target_link_libraries(Degrees PRIVATE Complete_Graph)
add_custom_library(Edge_Limits SHARED)
target_link_libraries(Edge_Limits PRIVATE cppitertools)


add_library(Metrics INTERFACE)
target_link_libraries(Metrics INTERFACE Degrees Edge_Limits Math)
