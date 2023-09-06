#ifndef SYCL_GRAPH_GENERATION_RANDOM_CONNECT_USM_INL_HPP
#define SYCL_GRAPH_GENERATION_RANDOM_CONNECT_USM_INL_HPP
#include <Sycl_Graph/Generation/Complete_Graph.hpp>
#include <Sycl_Graph/Metrics/Edge_Limits.hpp>
#include <Sycl_Graph/Utils/Buffer_Utils.hpp>
#include <Sycl_Graph/Utils/Math_Inl.hpp>
#include <Sycl_Graph/Utils/work_groups.hpp>
#include <oneapi/dpl/random>
namespace Sycl_Graph::USM {
  template <typename RNG>
  auto random_connect_kernel(sycl::handler& h, uint32_t* from_acc, uint32_t* to_acc, RNG* rng_acc,
                             Edge_t* e_acc, uint32_t* count_acc, uint32_t N_from, uint32_t N_to,
                             float p, uint32_t N_edges_max, auto nd_range) {
    auto N_per_work_item = get_N_per_work_item(N_edges_max, nd_range);
    h.parallel_for(nd_range, [=](sycl::nd_item<1> it) {
      auto gid = it.get_global_id();
      auto& rng = rng_acc[gid];
      oneapi::dpl::bernoulli_distribution dist(p);
      auto count = 0;
      for (auto e_idx = gid * N_per_work_item; e_idx < (gid + 1) * N_per_work_item; ++e_idx) {
        if (e_idx > N_edges_max) break;
        if (dist(rng)) {
          auto from_idx = floor_div(e_idx, N_from);
          auto to_idx = e_idx % N_to;

          e_acc[count] = Edge_t(from_acc[from_idx], to_acc[to_idx]);
          count++;
        }
      }
      count_acc[gid] = count;
    });
  }

  struct Merge_Edge_Vectors {
    Merge_Edge_Vectors(Edge_t* p_edges, uint32_t* N_edges_tot, uint32_t N_global, uint32_t* N_edges)
        : e_acc(p_edges), N_global(N_global), N_edges_tot(N_edges_tot), N_edges(N_edges) {}
    void operator()() const {
      auto N_merged_edges = 0;
      auto offset = 0;
      for (int i = 0; i < N_global; i++) {
        for (int j = 0; j < N_edges[i]; j++) {
          e_acc[N_merged_edges] = e_acc[offset + j];
          N_merged_edges++;
        }
        offset += N_edges[i];
      }
      N_edges_tot[0] = N_merged_edges;
    }

    const uint32_t N_global;

  private:
    Edge_t* e_acc;
    uint32_t* N_edges_tot;
    uint32_t* N_edges;
  };

  template <typename RNG>
  sycl::event random_connect(sycl::queue& q, uint32_t* from, uint32_t* to, RNG* rngs, const float p,
                             uint32_t N_from, uint32_t N_to, uint32_t N_rngs, uint32_t N_edges_max,
                             Edge_t* edges, uint32_t* N_edges_tot,
                             std::vector<sycl::event> dep_events = {}) {
    if (p == 0) {
      N_edges_tot[0] = 0;
      return {};
    }
    auto p_N_edges = sycl::malloc_shared<uint32_t>(N_rngs, q);
    auto nd_range = get_nd_range(q, N_rngs);
    auto sample_event = q.submit([&](sycl::handler& h) {
      h.depends_on(dep_events);
      random_connect_kernel(h, from, to, rngs, edges, p_N_edges, N_from, N_to, p, N_edges_max, nd_range);
    });
    sample_event.wait();
    auto sort_event = q.submit([&](sycl::handler& h) {
      h.depends_on(sample_event);
      h.single_task(Merge_Edge_Vectors(edges, N_edges_tot, N_rngs, p_N_edges));
    });
    q.wait();
    sycl::free(p_N_edges, q);
    return sort_event;
  }

  template <typename RNG>
  std::vector<Edge_t> random_connect(sycl::queue& q, const std::vector<uint32_t>& from,
                                     const std::vector<uint32_t>& to,
                                     const std::vector<uint32_t>& seeds, const auto p) {
    if (p == 0) {
      return {};
    }

    std::vector<RNG> rngs;
    rngs.reserve(seeds.size());
    for (auto seed : seeds) {
      rngs.emplace_back(RNG(seed));
    }
    std::vector<sycl::event> init_events(3);
    auto p_from = initialize_device_usm(from, q, init_events[0]);
    auto p_to = initialize_device_usm(to, q, init_events[1]);
    auto p_rng = initialize_device_usm(rngs, q, init_events[2]);
    auto N_edges_max = bipartite_graph_max_edges(from.size(), to.size());
    auto p_edges = sycl::malloc_device<Edge_t>(N_edges_max, q);
    auto p_N_edges_tot = sycl::malloc_shared<uint32_t>(1, q);

    auto event
        = random_connect<RNG>(q, p_from, p_to, p_rng, p, from.size(), to.size(), seeds.size(),
                              N_edges_max, p_edges, p_N_edges_tot, init_events);
    std::vector<Edge_t> edges(p_N_edges_tot[0]);
    q.memcpy(edges.data(), p_edges, sizeof(Edge_t)*p_N_edges_tot[0], event).wait();
    return edges;
  }

  template <typename RNG> std::vector<Edge_t> self_connect(sycl::queue& q,
                                                           const std::vector<uint32_t> ids,
                                                           const std::vector<uint32_t>& seeds,
                                                           const auto p, bool self_loops) {
    auto edgelist = random_connect<RNG>(q, ids, ids, seeds, p);
    if (!self_loops) {
      edgelist.erase(std::remove_if(edgelist.begin(), edgelist.end(),
                                    [](auto e) { return e.first == e.second; }),
                     edgelist.end());
    }
    return edgelist;
  }

}  // namespace Sycl_Graph::USM

#endif
