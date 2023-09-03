#ifndef SYCL_GRAPH_GENERATION_COMPLETE_GRAPH_HPP
#define SYCL_GRAPH_GENERATION_COMPLETE_GRAPH_HPP

#include <Sycl_Graph/Graph/Graph.hpp>

std::vector<Edge_t> complete_graph(std::size_t N_vertices, bool directed, bool self_loops);


#endif
