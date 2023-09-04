#ifndef SYCL_GRAPH_GENERATION_SBM_USM_INL_HPP
#define SYCL_GRAPH_GENERATION_SBM_USM_INL_HPP
#include <Sycl_Graph/Generation/Random_Connect.hpp>
#include <Sycl_Graph/Generation/SBM/SBM_Types.hpp>
#include <Sycl_Graph/Metrics/Edge_Limits.hpp>
#include <Sycl_Graph/Utils/RNG_Generation.hpp>
#include <Sycl_Graph/Utils/work_groups.hpp>
namespace Sycl_Graph::USM {

  template <typename RNG>
  sycl::event generate_SBM(sycl::queue& q, const std::shared_ptr<uint32_t>& vertices,
                           const std::vector<uint32_t>& community_sizes,
                           const std::vector<std::vector<float>>& p_mat, const std::shared_ptr<RNG>& rngs,
                           uint32_t N_rngs, const std::shared_ptr<Edge_t>& edges,
                           const std::shared_ptr<uint32_t>& N_edges,
                           uint32_t N_edges_max = std::numeric_limits<uint32_t>::max()) {
    auto N_communities = community_sizes.size();
    auto N_connections = complete_graph_max_edges(N_communities);
    auto nd_range = get_nd_range(q, N_rngs);
    std::vector<sycl::event> sample_events(N_connections);
    auto edge_offset = 0;
    std::vector<Edge_t> ccm = complete_graph(N_communities);

    std::vector<uint32_t> community_offsets(N_communities);
    std::partial_sum(community_sizes.begin(), community_sizes.end(), community_offsets.begin());

    // Distribute edge allocated space across connections (if too low)
    auto max_connection_edges = SBM_distributed_max_edges(community_sizes, N_edges_max);
    std::vector<uint32_t> edge_offsets(N_connections);
    std::partial_sum(max_connection_edges.begin(), max_connection_edges.end(),
                     edge_offsets.begin());

    using p_u32_t = std::shared_ptr<uint32_t>;
    sycl::event sort_event;
    // Random connect between all communities
    std::for_each(ccm.begin(), ccm.end(), [=, con_idx = 0](const Edge_t connection) mutable {
      auto p = p_mat[connection.from()][connection.to()];
      sample_events[con_idx] = random_connect(
          q, std::forward<const p_u32_t>(shared_offset(vertices, community_offsets[connection.from()])),
          std::forward<const p_u32_t>(shared_offset(vertices, community_offsets[connection.to()])),
          std::forward<const std::shared_ptr<RNG>>(rngs), p, community_sizes[connection.from()],
          community_sizes[connection.to()], N_rngs, max_connection_edges[con_idx],
          std::forward<const std::shared_ptr<Edge_t>>(shared_offset(edges, edge_offsets[con_idx])),
          std::forward<const p_u32_t>(shared_offset(N_edges, con_idx)));
      con_idx++;
    });


    // // Merge generated edges
    // auto N_edges_tot = make_shared_usm<uint32_t>(q, 1);
    // auto N_per_work_item = get_N_per_work_item(N_edges_max, nd_range);
    // auto sort_event = q.submit([&](sycl::handler& h) {
    //   h.depends_on(sample_events);
    //   h.single_task(Merge_Edge_Vectors(
    //       std::forward<const std::shared_ptr<Edge_t>>(edges), std::forward<const p_u32_t>(N_edges),
    //       std::forward<const p_u32_t>(N_edges_tot), N_rngs, N_per_work_item));
    // });

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
    auto N_edges_tot = make_shared_usm<uint32_t>(q, 1);
    auto edges = make_shared_usm<Edge_t>(q, N_edges_max);
    std::vector<uint32_t> v_idx(N_vertices);
    std::iota(v_idx.begin(), v_idx.end(), 0);
    auto p_vertices = make_shared_usm<uint32_t>(q, v_idx);
    auto N_edges = make_shared_usm<uint32_t>(q, N_connections);
    auto rngs = generate_shared_usm_rngs<RNG>(q, N_rngs, seed);

    auto event = generate_SBM(q, p_vertices, community_sizes, p_mat, rngs, N_rngs, edges, N_edges,
                              N_edges_max);

    return read_SBM_graph(edges, N_edges, N_connections, v_idx, community_sizes);
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
