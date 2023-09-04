#include <Sycl_Graph/Generation/SBM/SBM_Types.hpp>
#include <Sycl_Graph/Utils/Buffer_Utils.hpp>
namespace Sycl_Graph {

  std::vector<Edge_t> SBM_Graph_t::get_flat_edges() const {
    std::vector<Edge_t> flat_edges;
    for (auto&& e : edges) {
      flat_edges.insert(flat_edges.end(), e.begin(), e.end());
    }
    return flat_edges;
  }
  std::vector<uint32_t> SBM_Graph_t::get_flat_vertices() const {
    std::vector<uint32_t> flat_vertices;
    for (auto&& v : vertices) {
      flat_vertices.insert(flat_vertices.end(), v.begin(), v.end());
    }
    return flat_vertices;
  }
  std::size_t SBM_Graph_t::get_N_edges() const {
    return std::accumulate(
        edges.begin(), edges.end(), 0,
        [](const uint32_t sum, const std::vector<Edge_t>& e) { return sum + e.size(); });
  }
  std::size_t SBM_Graph_t::get_N_vertices() const {
    return std::accumulate(
        vertices.begin(), vertices.end(), 0,
        [](const uint32_t sum, const std::vector<uint32_t>& v) { return sum + v.size(); });
  }

  std::size_t SBM_Graph_t::get_N_communities() const { return vertices.size(); }

  std::size_t SBM_Graph_t::get_N_connections() const { return edges.size(); }

  SBM_Graph_t read_SBM_graph(std::shared_ptr<Edge_t>& p_edge, std::shared_ptr<uint32_t>& N_edges,
                             uint32_t N_connections, const std::vector<uint32_t>& p_vertices,
                             const std::vector<uint32_t>& community_sizes) {
    auto N_communities = community_sizes.size();
    std::vector<std::vector<Edge_t>> edges(N_connections);
    auto edge_offsets = USM::shared_usm_partial_sum(N_edges, N_connections);
    std::vector<uint32_t> vertex_offsets(N_communities, 0);
    std::partial_sum(community_sizes.begin(), community_sizes.end(), vertex_offsets.begin());
    for (int con_idx = 0; con_idx < N_connections; con_idx++) {
      edges[con_idx]
          = std::vector<Edge_t>(p_edge.get() + edge_offsets[con_idx],
                                p_edge.get() + edge_offsets[con_idx] + N_edges.get()[con_idx]);
    }
    std::vector<std::vector<uint32_t>> vertices(N_communities);
    for (int c_idx = 0; c_idx < N_communities; c_idx++) {
      vertices[c_idx]
          = std::vector<uint32_t>(p_vertices.data() + vertex_offsets[c_idx],
                                  p_vertices.data() + vertex_offsets[c_idx] + community_sizes[c_idx]);
    }

    return {edges, vertices};
  }
}  // namespace Sycl_Graph
