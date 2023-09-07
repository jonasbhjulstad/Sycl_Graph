// #include <Sycl_Graph/Generation/SBM.hpp>
#include <Sycl_Graph/Generation/Random_Connect/Random_Connect_USM_Inl.hpp>
#include <Sycl_Graph/Generation/SBM/SBM_Types.hpp>
#include <Sycl_Graph/Metrics/Edge_Limits.hpp>
#include <Sycl_Graph/Utils/RNG_Generation.hpp>
#include <oneapi/dpl/random>
namespace Sycl_Graph::USM {


  template <typename RNG>
  sycl::event generate_SBM(sycl::queue& q, const SBM_Param_t& param, SBM_Device_USM_t<RNG>& sbm_usm,
                           const std::vector<std::vector<float>>& p_mat, bool directed = false) {
    auto nd_range = get_nd_range(q, sbm_usm.N_rngs);

    std::vector<sycl::event> sample_events(param.N_connections);

    // Random connect over all connections in connection community map (ccm)
    std::transform(param.ccm.begin(), param.ccm.end(), sample_events.begin(),
                   [&, i = 0](auto connection) mutable {
                     auto p_vertex_from = sbm_usm.vertices + param.community_offsets[connection.from()];
                     auto p_vertex_to = sbm_usm.vertices + param.community_offsets[connection.to()];
                     auto p_edge = sbm_usm.edges + param.edge_offsets[i];
                     auto p = p_mat[connection.from()][connection.to()];
                     auto N_from = param.community_sizes[connection.from()];
                     auto N_to = param.community_sizes[connection.to()];
                     auto N_edges_max = param.edge_sizes[i];
                     auto p_N_edges = sbm_usm.N_edges + i;
                     i++;
                     return random_connect(q, p_vertex_from, p_vertex_to, sbm_usm.rngs, p, N_from, N_to,
                                           sbm_usm.N_rngs, N_edges_max, p_edge, p_N_edges, sbm_usm.init_events);
                   });
    q.wait();
    auto sort_event = q.submit([&](sycl::handler& h) {
      h.depends_on(sample_events);
      h.single_task(Merge_Vectors<Edge_t>(sbm_usm.edges, sbm_usm.N_edges_tot, param.N_connections, sbm_usm.N_edges));
    });
    return sort_event;
  }
  template <typename RNG> SBM_Graph_t generate_SBM(sycl::queue& q, const SBM_Param_t& sbm_param,
                                                   const std::vector<std::vector<float>>& p_mat,
                                                   bool directed = false,
                                                   uint32_t N_rngs = 0,
                                                   uint32_t seed = 23) {
    N_rngs = (N_rngs == 0) ? get_wg_size(q) : N_rngs;

    SBM_Device_USM_t<RNG> sbm_usm(sbm_param, N_rngs, seed, q);
    sbm_usm.wait();
    auto event
        = generate_SBM<RNG>(q, sbm_param, sbm_usm, p_mat, directed);
    auto sbm_data = sbm_usm.get_data();
    sbm_usm.free();
    SBM_Graph_t graph(sbm_param, sbm_data);
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
                                   uint32_t N_rngs = 0) {
    SBM_Param_t SBM_param(N_pop, N_communities, directed);
    auto p_mat = planted_SBM_p_mat(p_in, p_out, N_communities);
    return generate_SBM<RNG>(q, SBM_param, p_mat, directed, N_rngs, seed);
  }

}  // namespace Sycl_Graph::USM

int main() {
  // using namespace Sycl_Graph;
  // using namespace Sycl_Graph::USM;
  using RNG = oneapi::dpl::ranlux24;
  sycl::queue q(sycl::cpu_selector_v);
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
