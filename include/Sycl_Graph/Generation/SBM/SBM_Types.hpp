#ifndef SYCL_GRAPH_GENERATION_SBM_TYPES_HPP
#define SYCL_GRAPH_GENERATION_SBM_TYPES_HPP
#include <Sycl_Graph/Generation/SBM/SBM_Types.hpp>
#include <Sycl_Graph/Graph/Graph.hpp>
#include <Sycl_Graph/Utils/Buffer_Utils.hpp>
namespace Sycl_Graph {

  struct SBM_Graph_t {
    std::vector<std::vector<Edge_t>> edges;
    std::vector<std::vector<uint32_t>> vertices;
    std::vector<Edge_t> get_flat_edges() const {
      std::vector<Edge_t> flat_edges;
      for (auto&& e : edges) {
        flat_edges.insert(flat_edges.end(), e.begin(), e.end());
      }
      return flat_edges;
    }
    std::vector<uint32_t> get_flat_vertices() const {
      std::vector<uint32_t> flat_vertices;
      for (auto&& v : vertices) {
        flat_vertices.insert(flat_vertices.end(), v.begin(), v.end());
      }
      return flat_vertices;
    }
    std::size_t get_N_edges() const {
      return std::accumulate(
          edges.begin(), edges.end(), 0,
          [](const uint32_t sum, const std::vector<Edge_t>& e) { return sum + e.size(); });
    }
    std::size_t get_N_vertices() const {
      return std::accumulate(
          vertices.begin(), vertices.end(), 0,
          [](const uint32_t sum, const std::vector<uint32_t>& v) { return sum + v.size(); });
    }

    std::size_t get_N_communities() const { return vertices.size(); }

    std::size_t get_N_connections() const { return edges.size(); }

    bool is_edges_valid() const
    {
      return std::all_of(edges.begin(), edges.end(), [](const auto& e_list)
      {
        return std::all_of(e_list.begin(), e_list.end(), [](auto e)
        {
          return e.is_valid();
        });
      });
    }

    bool is_vertices_valid() const
    {
      return std::all_of(vertices.begin(), vertices.end(), [](const auto& v_list)
      {
        return std::all_of(v_list.begin(), v_list.end(), [](auto v)
        {
          return v != std::numeric_limits<uint32_t>::max();
        });
      });
    }
    bool is_valid() const{
      return is_edges_valid() && is_vertices_valid();
    }
  };
  SBM_Graph_t read_SBM_graph(Edge_t* p_edge, uint32_t* p_N_edges, uint32_t N_connections,
                             const std::vector<uint32_t>& vertices_flat,
                             const std::vector<uint32_t>& community_sizes, sycl::queue& q, std::vector<sycl::event> dep_events = {}) {
    auto N_communities = community_sizes.size();
    std::vector<uint32_t> N_edges(N_connections);
    q.memcpy(N_edges.data(), p_N_edges, N_connections * sizeof(uint32_t), dep_events).wait();
    std::vector<uint32_t> edge_offsets(N_connections, 0);
    std::partial_sum(N_edges.begin(), N_edges.end() - 1, edge_offsets.begin() + 1);
    std::vector<std::vector<Edge_t>> edges(N_connections);
    std::vector<uint32_t> vertex_offsets(N_communities, 0);
    std::partial_sum(community_sizes.begin(), community_sizes.end()-1, vertex_offsets.begin()+1);
    std::vector<sycl::event> edge_cpy_events(N_connections);
    auto N_edges_tot = std::accumulate(N_edges.begin(), N_edges.end(), 0);
    std::vector<Edge_t> edges_flat(N_edges_tot);
    q.memcpy(edges_flat.data(), p_edge, N_edges_tot * sizeof(Edge_t), dep_events).wait();
    std::transform(edge_offsets.begin(), edge_offsets.end(), N_edges.begin(), edges.begin(), [&edges_flat](auto offset, auto N_edge)
    {
      return std::vector<Edge_t>(edges_flat.begin() + offset,  edges_flat.begin() + offset + N_edge);
    });

    std::vector<std::vector<uint32_t>> vertices(N_communities);
    for (int c_idx = 0; c_idx < N_communities; c_idx++) {
      vertices[c_idx] = std::vector<uint32_t>(
          vertices_flat.data() + vertex_offsets[c_idx],
          vertices_flat.data() + vertex_offsets[c_idx] + community_sizes[c_idx]);
    }
    return {edges, vertices};
  }
}  // namespace Sycl_Graph

#endif
