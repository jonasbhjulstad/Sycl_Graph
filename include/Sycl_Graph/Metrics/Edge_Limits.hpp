#ifndef SYCL_GRAPH_METRICS_EDGE_LIMITS_HPP
#define SYCL_GRAPH_METRICS_EDGE_LIMITS_HPP
#include <cmath>
#include <vector>
#include <cstdint>
namespace Sycl_Graph
{
std::size_t complete_graph_max_edges(std::size_t N_vertices, bool directed = false, bool self_loops = true);

std::size_t complete_digraph_max_edges(std::size_t N_vertices, bool self_loops = true);

std::size_t bipartite_graph_max_edges(std::size_t N_vertices_0, std::size_t N_vertices_1, bool directed = false);

std::size_t SBM_expected_edges(const std::vector<uint32_t>& community_sizes, const std::vector<float>& p, bool directed = false, bool self_loops = true);
std::vector<std::size_t> SBM_max_connection_edges(const std::vector<uint32_t>& commmunity_sizes);

std::vector<std::size_t> SBM_distributed_max_edges(const std::vector<uint32_t>& community_sizes, uint32_t N_max_edges);

}  // namespace Sycl_Graph
#endif
