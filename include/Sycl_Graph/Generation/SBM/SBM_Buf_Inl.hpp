#ifndef SYCL_GRAPH_GENERATION_SBM_BUF_INL_HPP
#define SYCL_GRAPH_GENERATION_SBM_BUF_INL_HPP
#include <Sycl_Graph/Generation/Random_Connect.hpp>
#include <Sycl_Graph/Metrics/Edge_Limits.hpp>
namespace Sycl_Graph {
  template <typename RNG> sycl::event generate_SBM(sycl::queue& q,
                                                    sycl::buffer<uint32_t, 1>& from,
                                                   sycl::buffer<uint32_t, 1>& to,
                                                   const std::vector<uint32_t>& community_sizes,
                                                   const std::vector<float>& p,
                                                   sycl::buffer<RNG>& rngs, Edgebuf_t<1>& edges,
                                                   sycl::buffer<uint32_t>& N_edges_tot)
    {
        auto N_communities = from.get_range()[0];
        auto N_connections = complete_graph_max_edges(N_communities);
        auto N_edges_max = edges.get_range()[0];
        if (N_edges_max < SBM_expected_edges(community_sizes, p, directed, self_loops))
        {
            throw std::runtime_error("Space allocated in edge buffer is less than the expected number of edges.");
        }

        std::vector<sycl::event> rng_events(N_connections);
        auto edge_offset = 0;
        for(auto i = 0; i < N_connections; i++)
        {
            auto from_acc = sycl::accessor<uint32_t, 1, sycl::access::mode::read>(from, sycl::range<1>(community_sizes[i]), sycl::range<1>(edge_offset));
            auto to_acc = sycl::accessor<uint32_t, 1, sycl::access::mode::read>(to, sycl::range<1>(community_sizes[i]), sycl::range<1>(edge_offset));
            auto rng_acc = rngs.template get_access<sycl::access::mode::read_write>(h);

        }
    }
}  // namespace Sycl_Graph

#endif
