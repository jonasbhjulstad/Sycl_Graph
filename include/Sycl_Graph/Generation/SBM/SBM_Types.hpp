#ifndef SYCL_GRAPH_GENERATION_SBM_TYPES_HPP
#define SYCL_GRAPH_GENERATION_SBM_TYPES_HPP
#include <Sycl_Graph/Graph/Graph.hpp>
namespace Sycl_Graph {

  struct SBM_Graph_t {
    std::vector<std::vector<Edge_t>> edges;
    std::vector<std::vector<uint32_t>> vertices;
    std::vector<Edge_t> get_flat_edges() const;
    std::vector<uint32_t> get_flat_vertices() const;
    std::size_t get_N_edges() const;
    std::size_t get_N_vertices() const;
    std::size_t get_N_communities() const;
    std::size_t get_N_connections() const;
  };

    SBM_Graph_t read_SBM_graph(std::shared_ptr<Edge_t>& p_edge, std::shared_ptr<uint32_t>& N_edges,
                             uint32_t N_connections, const std::vector<uint32_t>& p_vertices,
                             const std::vector<uint32_t>& community_sizes);
}  // namespace Sycl_Graph
#endif
