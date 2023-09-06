#ifndef SYCL_GRAPH_GENERATION_SBM_USM_INL_HPP
#define SYCL_GRAPH_GENERATION_SBM_USM_INL_HPP
#include <Sycl_Graph/Generation/Random_Connect.hpp>
#include <Sycl_Graph/Generation/SBM/SBM_Types.hpp>
#include <Sycl_Graph/Metrics/Edge_Limits.hpp>
#include <Sycl_Graph/Utils/RNG_Generation.hpp>
#include <Sycl_Graph/Utils/work_groups.hpp>
namespace Sycl_Graph::USM {

  template <typename RNG>
  sycl::event generate_SBM(sycl::queue& q, uint32_t* vertices,
                           const std::vector<uint32_t>& community_sizes,
                           const std::vector<std::vector<float>>& p_mat, RNG* rngs, uint32_t N_rngs,
                           Edge_t* edges, uint32_t* N_edges,
                           uint32_t N_edges_max = std::numeric_limits<uint32_t>::max()) {
    auto N_communities = community_sizes.size();
    auto N_connections = complete_graph_max_edges(N_communities, false, true);
    auto nd_range = get_nd_range(q, N_rngs);
    std::vector<sycl::event> sample_events(N_connections);
    auto edge_offset = 0;
    std::vector<Edge_t> ccm = complete_graph(N_communities, false, true);

    std::vector<uint32_t> community_offsets(N_communities);
    for (int i = 0; i < N_communities; i++) {
      community_offsets[i]
          = std::accumulate(community_sizes.begin(), community_sizes.begin() + i, 0);
    }
    // Distribute edge allocated space across connections (if too low)
    auto max_connection_edges = SBM_distributed_max_edges(community_sizes, N_edges_max);
    std::vector<uint32_t> edge_offsets(N_connections);
    std::partial_sum(max_connection_edges.begin(), max_connection_edges.end(),
                     edge_offsets.begin());

    using p_u32_t = std::shared_ptr<uint32_t>;
    // sycl::event sort_event;
    // Random connect between all communities
    // std::for_each(ccm.begin(), ccm.end(), [&, con_idx = 0](const Edge_t connection) mutable {
    //   auto p = p_mat[connection.from()][connection.to()];
    //   auto p_from = vertices + community_offsets[connection.from()];
    //   auto p_to = vertices + community_offsets[connection.to()];
    //   auto p_edges = edges + edge_offsets[con_idx];
    //   auto p_N_edges = N_edges + con_idx;

    //   sample_events[con_idx]
    //       = random_connect(q, p_from, p_to, rngs, p, community_sizes[connection.from()],
    //                        community_sizes[connection.to()], N_rngs,
    //                        max_connection_edges[con_idx], p_edges, p_N_edges);
    //   con_idx++;
    // });
    sample_events[0]
        = random_connect(q, vertices, vertices, rngs, p_mat[0][0], community_sizes[0],
                         community_sizes[0], N_rngs, max_connection_edges[0], edges, N_edges);

    // Merge generated edges
    auto sort_event = q.submit([&](sycl::handler& h) {
      h.depends_on(sample_events);
      h.single_task(Merge_Edge_Vectors(edges, N_edges, N_connections, N_edges));
    });
    return sort_event;
  }

  template <typename RNG> SBM_Graph_t generate_SBM(sycl::queue& q,
                                                   const std::vector<uint32_t>& community_sizes,
                                                   const std::vector<std::vector<float>>& p_mat,
                                                   uint32_t N_rngs, uint32_t seed) {
    auto N_communities = community_sizes.size();
    auto N_vertices = std::accumulate(community_sizes.begin(), community_sizes.end(), 0);
    auto N_connections = complete_graph_max_edges(N_communities);
    auto N_edges_max = complete_graph_max_edges(N_vertices);
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
    auto event = generate_SBM(q, p_vertices, community_sizes, p_mat, rngs, N_rngs, edges, N_edges,
                              N_edges_max);
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
                                   float p_in, float p_out, uint32_t seed, uint32_t N_rng = 0) {
    auto N_wg = (N_rng == 0) ? get_wg_size(q) : N_rng;
    std::vector<uint32_t> community_sizes(N_communities, N_pop);
    auto p_mat = planted_SBM_p_mat(p_in, p_out, N_communities);
    return generate_SBM<RNG>(q, community_sizes, p_mat, N_wg, seed);
  }

}  // namespace Sycl_Graph::USM

#endif
