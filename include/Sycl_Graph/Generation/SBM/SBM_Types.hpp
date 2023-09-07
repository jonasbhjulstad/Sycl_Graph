#ifndef SYCL_GRAPH_GENERATION_SBM_TYPES_HPP
#define SYCL_GRAPH_GENERATION_SBM_TYPES_HPP
#include <Sycl_Graph/Generation/SBM/SBM_Types.hpp>
#include <Sycl_Graph/Graph/Graph.hpp>
#include <Sycl_Graph/Utils/Buffer_Utils.hpp>
#include <Sycl_Graph/Utils/RNG_Generation.hpp>
#include <Sycl_Graph/Generation/Complete_Graph.hpp>
#include <Sycl_Graph/Metrics/Edge_Limits.hpp>
namespace Sycl_Graph {

  struct SBM_Param_t {
    SBM_Param_t(const std::vector<uint32_t>&& community_sizes, bool directed = false)
        : N_communities{community_sizes.size()},
          N_vertices{static_cast<std::size_t>(std::accumulate(community_sizes.begin(), community_sizes.end(), 0))},
          N_connections{complete_graph_max_edges(community_sizes.size(), directed, true)},
          vertex_indices{iota(N_vertices)},
          community_sizes{community_sizes},
          community_offsets{get_offsets(community_sizes)},
          ccm{complete_graph(N_communities, directed, true)} {
            edge_sizes.resize(N_connections);
            std::transform(ccm.begin(), ccm.end(), edge_sizes.begin(),
                           [&, i = 0](const Edge_t connection) mutable {
                            return bipartite_graph_max_edges(community_sizes[connection.from()],
                                                             community_sizes[connection.to()], false);
                           });
            edge_offsets = get_offsets(edge_sizes);
            N_edges = std::accumulate(edge_sizes.begin(), edge_sizes.end(), 0);
          }
    SBM_Param_t(uint32_t N_pop, uint32_t N_communities, bool directed = false)
        : SBM_Param_t(std::vector<uint32_t>(N_communities, N_pop), directed) {}

    std::size_t N_vertices;
    std::size_t N_edges;
    std::size_t N_connections;
    std::size_t N_communities;
    std::vector<uint32_t> vertex_indices;
    std::vector<uint32_t> community_sizes;
    std::vector<uint32_t> community_offsets;
    std::vector<uint32_t> edge_sizes;
    std::vector<uint32_t> edge_offsets;
    std::vector<Edge_t> ccm;

    std::vector<std::vector<uint32_t>> community_vertex_lists() const {
      std::vector<std::vector<uint32_t>> vertex_lists(N_communities);
      for (int i = 0; i < N_communities; i++) {
        vertex_lists[i].reserve(community_sizes[i]);
        for (int j = 0; j < community_sizes[i]; j++) {
          vertex_lists[i].push_back(vertex_indices[community_offsets[i] + j]);
        }
      }
      return vertex_lists;
    }

  private:
    std::vector<uint32_t> get_offsets(const std::vector<uint32_t>& vec) {
      std::vector<uint32_t> offsets(vec.size());
      std::partial_sum(vec.begin(), vec.end() - 1, offsets.begin() + 1);
      return offsets;
    };
    std::vector<uint32_t> iota(std::size_t N) {
      std::vector<uint32_t> v(N);
      std::iota(v.begin(), v.end(), 0);
      return v;
    }
  };

  struct SBM_Data_t {
    std::vector<uint32_t> vertices;
    std::vector<Edge_t> edges;
    std::vector<uint32_t> N_edges;
    std::vector<uint32_t> N_edges_tot;

    std::vector<std::vector<Edge_t>> edge_connection_lists() const {
      auto edge_offsets = this->edge_offsets();
      std::vector<std::vector<Edge_t>> edge_lists(N_edges.size());
      for (int i = 0; i < N_edges.size(); i++) {
        edge_lists[i].reserve(N_edges[i]);
        for (int j = 0; j < N_edges[i]; j++) {
          edge_lists[i].push_back(edges[edge_offsets[i] + j]);
        }
      }
      return edge_lists;
    }

  private:
    std::vector<uint32_t> edge_offsets() const {
      std::vector<uint32_t> offsets(N_edges.size());
      std::partial_sum(N_edges.begin(), N_edges.end() - 1, offsets.begin() + 1);
      return offsets;
    }
  };

  struct SBM_Graph_t {
    SBM_Graph_t(const SBM_Param_t& p, const SBM_Data_t& data)
        : vertices{p.community_vertex_lists()}, edges{data.edge_connection_lists()} {}
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

namespace Sycl_Graph::USM {
  template <typename RNG> struct SBM_Device_USM_t {
    SBM_Device_USM_t(const SBM_Param_t& param, uint32_t N_rngs, uint32_t seed, sycl::queue& q)
        : N_rngs(N_rngs), init_events(2), read_events(4), q(q), N_connections(param.N_connections), N_vertices(param.N_vertices) {
      rngs = generate_usm_rngs<RNG>(q, N_rngs, seed, init_events[0]);
      vertices = initialize_device_usm(param.vertex_indices, q, init_events[1]);
      edges = sycl::malloc_device<Edge_t>(param.N_edges, q);
      N_edges = sycl::malloc_shared<uint32_t>(param.N_connections, q);
      N_edges_tot = sycl::malloc_shared<uint32_t>(1, q);
    }

    const uint32_t N_rngs;
    RNG* rngs;
    uint32_t* vertices;
    Edge_t* edges;
    // Number of edges in each connection
    uint32_t* N_edges;
    // Total number of edges in the SBM
    uint32_t* N_edges_tot;
    uint32_t N_connections;
    uint32_t N_vertices;
    std::vector<sycl::event> init_events;
    std::vector<sycl::event> read_events;

    void free() {
      sycl::free(rngs, q);
      sycl::free(vertices, q);
      sycl::free(edges, q);
      sycl::free(N_edges, q);
      sycl::free(N_edges_tot, q);
    }
    std::vector<sycl::event> read_data() {
      events_wait(init_events);
      data.N_edges_tot.resize(1);
      data.N_edges.resize(N_connections);
      data.vertices.resize(N_vertices);
      read_events[3] = read_device_usm(data.N_edges_tot, N_edges_tot, q);
      read_events[2] = read_device_usm(data.N_edges, N_edges, q);
      events_wait(read_events);
      data.edges.resize(N_edges_tot[0]);
      read_events[0] = read_device_usm(data.vertices, vertices, q);
      read_events[1] = read_device_usm(data.edges, edges, q);
      data_read = true;
      return read_events;
    }

    void wait()
    {
      events_wait(init_events);
      events_wait(read_events);
    }

    SBM_Data_t get_data() {
      if (!data_read) read_data();
      events_wait(read_events);
      return data;
    }

  private:
    bool data_read = false;
    SBM_Data_t data;
    sycl::queue& q;
  };
}  // namespace Sycl_Graph::USM

#endif
