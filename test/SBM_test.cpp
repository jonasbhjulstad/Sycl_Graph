// #include <Sycl_Graph/Generation/SBM.hpp>
#include <Sycl_Graph/Generation/Random_Connect/Random_Connect_USM_Inl.hpp>
#include <Sycl_Graph/Generation/SBM/SBM_Types.hpp>
#include <Sycl_Graph/Metrics/Edge_Limits.hpp>
#include <Sycl_Graph/Utils/RNG_Generation.hpp>
#include <oneapi/dpl/random>
namespace Sycl_Graph::USM {

  template <typename RNG> sycl::event generate_SBM(sycl::queue& q, uint32_t* vertices,
                                                   const std::vector<uint32_t>& community_sizes,
                                                   const std::vector<std::vector<float>>& p_mat,
                                                   RNG* rngs, uint32_t N_rngs, Edge_t* edges,
                                                   uint32_t* N_edges, bool directed = false) {
    auto N_communities = community_sizes.size();
    auto N_connections = complete_graph_max_edges(N_communities, directed, true);
    auto nd_range = get_nd_range(q, N_rngs);
    auto get_offsets = [](const std::vector<uint32_t>& vec) {
      std::vector<uint32_t> offsets(vec.size());
      std::partial_sum(vec.begin(), vec.end() - 1, offsets.begin() + 1);
      return offsets;
    };

    std::vector<sycl::event> sample_events(N_connections);
    auto edge_offset = 0;
    std::vector<Edge_t> ccm = complete_graph(N_communities, directed, true);

    auto community_offsets = get_offsets(community_sizes);

    // Maximum number of edges per connection

    std::vector<uint32_t> N_max_connection_edges(N_connections);
    std::transform(ccm.begin(), ccm.end(), N_max_connection_edges.begin(), [&](auto connection) {
      return bipartite_graph_max_edges(community_sizes[connection.from()],
                                       community_sizes[connection.to()], directed);
    });

    // Edge Offsets
    auto edge_offsets = get_offsets(N_max_connection_edges);

    // Random connect over all connections in connection community map (ccm)
    std::transform(ccm.begin(), ccm.end(), sample_events.begin(),
                   [&, i = 0](auto connection) mutable {
                     auto p_vertex_from = vertices + community_offsets[connection.from()];
                     auto p_vertex_to = vertices + community_offsets[connection.to()];
                     auto p_edges = edges + edge_offsets[i];
                     auto p = p_mat[connection.from()][connection.to()];
                     auto N_from = community_sizes[connection.from()];
                     auto N_to = community_sizes[connection.to()];
                     auto N_edges_max = N_max_connection_edges[i];
                     auto p_N_edges = N_edges + i;
                     i++;
                     return random_connect(q, p_vertex_from, p_vertex_to, rngs, p, N_from, N_to,
                                           N_rngs, N_edges_max, edges, p_N_edges);
                   });
    q.wait();
    auto p_N_edges_tot = sycl::malloc_shared<uint32_t>(1, q);
    auto sort_event = q.submit([&](sycl::handler& h) {
      h.depends_on(sample_events);
      h.single_task(Merge_Edge_Vectors(edges, p_N_edges_tot, N_connections, N_edges));
    });
    sort_event.wait();
    sycl::free(p_N_edges_tot, q);
    return sort_event;
  }
  template <typename RNG>
  SBM_Graph_t generate_SBM(sycl::queue& q, const std::vector<uint32_t>& community_sizes,
                           const std::vector<std::vector<float>>& p_mat, uint32_t N_rngs,
                           bool directed = false, uint32_t seed = 23) {
    auto N_communities = community_sizes.size();
    auto N_vertices = std::accumulate(community_sizes.begin(), community_sizes.end(), 0);
    auto N_connections = complete_graph_max_edges(N_communities, directed, true);
    auto N_edges_max = complete_graph_max_edges(N_vertices, directed, true);
    auto edges = sycl::malloc_device<Edge_t>(N_edges_max, q);
    std::vector<uint32_t> v_idx(N_vertices);
    std::iota(v_idx.begin(), v_idx.end(), 0);
    sycl::event v_event;
    auto p_vertices = initialize_device_usm(v_idx, q, v_event);
    auto N_edges = sycl::malloc_shared<uint32_t>(N_connections, q);
    sycl::event rng_event;
    auto rngs = generate_usm_rngs<RNG>(q, N_rngs, seed, rng_event);
    v_event.wait();
    rng_event.wait();
    auto event = generate_SBM<RNG>(q, p_vertices, community_sizes, p_mat, rngs, N_rngs, edges,
                                   N_edges, directed);
    sycl::free(p_vertices, q);
    sycl::free(rngs, q);
    auto graph = read_SBM_graph(edges, N_edges, N_connections, v_idx, community_sizes, q);
    sycl::free(edges, q);
    sycl::free(N_edges, q);
    return graph;
  }
  auto planted_SBM_p_mat(float p_in, float p_out, uint32_t N_communities) {
    std::vector<std::vector<float>> p_mat(N_communities, std::vector<float>(N_communities, p_out));
    // fill all vectors with p_out
    std::for_each(p_mat.begin(), p_mat.end(),
                  [p_out](std::vector<float>& v) { std::fill(v.begin(), v.end(), p_out); });
    for (int i = 0; i < N_communities; i++) {
      p_mat[i][i] = p_in;
    }

    return p_mat;
  }

  template <typename RNG>
  SBM_Graph_t generate_planted_SBM(sycl::queue& q, uint32_t N_pop, uint32_t N_communities,
                                   float p_in, float p_out, uint32_t seed, bool directed = false,
                                   uint32_t N_rng = 0) {
    auto N_wg = (N_rng == 0) ? get_wg_size(q) : N_rng;
    std::vector<uint32_t> community_sizes(N_communities, N_pop);
    auto p_mat = planted_SBM_p_mat(p_in, p_out, N_communities);
    return generate_SBM<RNG>(q, community_sizes, p_mat, N_wg, directed, seed);
  }

}  // namespace Sycl_Graph::USM

int main() {
  // using namespace Sycl_Graph;
  // using namespace Sycl_Graph::USM;
  using RNG = oneapi::dpl::ranlux24;

  sycl::queue q(sycl::gpu_selector_v);

  auto N_pop = 10;
  auto N_communities = 10;
  float p_in = 0.5;
  float p_out = 0.5;
  auto seed = 23;
  auto N_rngs = 16;
  auto graph = Sycl_Graph::USM::generate_planted_SBM<RNG>(q, N_pop, N_communities, p_in, p_out,
                                                          seed, true, N_rngs);
  assert(graph.is_valid());
  assert(graph.get_N_communities() == N_communities);
  assert(graph.get_N_connections()
         == Sycl_Graph::complete_graph_max_edges(N_communities, true, true));
  assert(graph.get_N_vertices() == N_pop * N_communities);

  p_in = 1.0f;
  p_out = 1.0f;
  graph = Sycl_Graph::USM::generate_planted_SBM<RNG>(q, N_pop, N_communities, p_in, p_out, seed,
                                                     true, N_rngs);
  assert(graph.is_valid());
  assert(graph.get_N_communities() == N_communities);
  assert(graph.get_N_connections()
         == Sycl_Graph::complete_graph_max_edges(N_communities, true, true));
  assert(graph.get_N_vertices() == N_pop * N_communities);
  assert(graph.get_N_edges()
         == Sycl_Graph::complete_graph_max_edges(N_pop * N_communities, true, true));

  // auto sbm_complete = generate_planted_SBM<RNG>(q, N_pop, N_communities, 1.0f, 1.0f, seed);

  return 0;
}
