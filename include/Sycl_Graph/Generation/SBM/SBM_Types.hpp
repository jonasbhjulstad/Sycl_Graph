#ifndef SYCL_GRAPH_GENERATION_SBM_TYPES_HPP
#define SYCL_GRAPH_GENERATION_SBM_TYPES_HPP
#include <Sycl_Graph/Generation/SBM/SBM_Types.hpp>
#include <Sycl_Graph/Graph/Graph.hpp>
#include <Sycl_Graph/Utils/Buffer_Utils.hpp>
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

    bool is_edges_valid() const;
    bool is_vertices_valid() const;
    bool is_valid() const;
  };
  SBM_Graph_t read_SBM_graph(Edge_t* p_edge, uint32_t* p_N_edges, uint32_t N_connections,
                             const std::vector<uint32_t>& vertices_flat,
                             const std::vector<uint32_t>& community_sizes, sycl::queue& q,
                             std::vector<sycl::event> dep_events = {});
}  // namespace Sycl_Graph

#endif
