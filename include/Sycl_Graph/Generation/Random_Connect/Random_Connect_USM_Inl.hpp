#ifndef SYCL_GRAPH_GENERATION_RANDOM_CONNECT_USM_INL_HPP
#define SYCL_GRAPH_GENERATION_RANDOM_CONNECT_USM_INL_HPP
#include <Sycl_Graph/Graph/Graph.hpp>
#include <Sycl_Graph/Metrics/Edge_Limits.hpp>
#include <Sycl_Graph/Utils/Buffer_Utils.hpp>
#include <Sycl_Graph/Utils/Math_Inl.hpp>
#include <Sycl_Graph/Utils/work_groups.hpp>
#include <oneapi/dpl/random>
namespace Sycl_Graph::USM {

  template <typename RNG> struct Random_Connect_Kernel {
    Random_Connect_Kernel(const std::shared_ptr<uint32_t>&& from_acc, const std::shared_ptr<uint32_t>&& to_acc,
                          const std::shared_ptr<RNG>&& rng_acc, const std::shared_ptr<Edge_t>&& e_acc,
                          const std::shared_ptr<uint32_t>&& count_acc, uint32_t N_from, uint32_t N_to,
                          float p, uint32_t Npw)
        : from_acc(from_acc.get()),
          to_acc(to_acc.get()),
          rng_acc(rng_acc.get()),
          e_acc(e_acc.get()),
          count_acc(count_acc.get()),
          N_from(N_from),
          N_to(N_to),
          p(p),
          N_per_work_item(Npw) {}

    void operator()(sycl::nd_item<1> it) const {
      auto gid = it.get_global_id();
      auto& rng = rng_acc[gid];
      oneapi::dpl::bernoulli_distribution dist(p);
      auto count = 0;
      for (auto e_idx = gid * N_per_work_item; e_idx < (gid + 1) * N_per_work_item; ++e_idx) {
        if (dist(rng)) {
          auto from_idx = floor_div(e_idx, N_from);
          auto to_idx = e_idx % N_to;
          e_acc[e_idx] = Edge_t(from_idx, to_idx);
          count++;
        }
      }
      count_acc[gid] = count;
    }
    const uint32_t N_from, N_to, N_per_work_item;
    const float p;

  private:
    uint32_t* from_acc;
    uint32_t* to_acc;
    RNG* rng_acc;
    Edge_t* e_acc;
    uint32_t* count_acc;
  };

  struct Merge_Edge_Vectors {
    Merge_Edge_Vectors(const std::shared_ptr<Edge_t>&& p_edges, const std::shared_ptr<uint32_t>&& p_count,
                       const std::shared_ptr<uint32_t>&& p_N_tot, uint32_t N_global, uint32_t Npw)
        : e_acc(p_edges.get()),
          count_acc(p_count.get()),
          Ntot_acc(p_N_tot.get()),
          N_global(N_global),
          N_per_work_item(Npw) {}
    void operator()() const {
      auto N_merged_edges = 0;
      for (int i = 0; i < N_global; i++) {
        for (int j = 0; j < N_per_work_item; j++) {
          e_acc[N_merged_edges] = e_acc[i * N_per_work_item + j];
          N_merged_edges++;
        }
      }
      Ntot_acc[0] = N_merged_edges;
    }

    const uint32_t N_per_work_item;
    const uint32_t N_global;

  private:
    Edge_t* e_acc;
    const uint32_t* count_acc;
    uint32_t* Ntot_acc;
  };

  template <typename RNG>
  sycl::event random_connect(sycl::queue& q, const std::shared_ptr<uint32_t>&& from,
                             const std::shared_ptr<uint32_t>&& to, const std::shared_ptr<RNG>&& rngs,
                             const float p, uint32_t N_from, uint32_t N_to, uint32_t N_rngs,
                             uint32_t N_edges_max, const std::shared_ptr<Edge_t>&& edges,
                             const std::shared_ptr<uint32_t>&& N_edges_tot) {
    if (p == 1) {
      N_edges_tot.get()[0] = bipartite_graph_max_edges(N_from, N_to);
      return {};
    } else if (p == 0) {
      N_edges_tot.get()[0] = 0;
      return {};
    }

    auto nd_range = get_nd_range(q, N_rngs);
    auto N_per_work_item = get_N_per_work_item(N_edges_max, nd_range);
    auto sample_event = q.submit([&](sycl::handler& h) {
      Random_Connect_Kernel<RNG> kernel(std::forward<const std::shared_ptr<uint32_t>>(from),
      std::forward<const std::shared_ptr<uint32_t>>(to),
      std::forward<const std::shared_ptr<RNG>>(rngs),
      std::forward<const std::shared_ptr<Edge_t>>(edges),
      std::forward<const std::shared_ptr<uint32_t>>(N_edges_tot), N_from, N_to, p, N_per_work_item);
      h.parallel_for(nd_range, kernel);
    });
    auto edge_counts = make_shared_usm<uint32_t>(q, N_rngs);
    auto sort_event = q.submit([&](sycl::handler& h) {
      h.depends_on(sample_event);
      h.single_task(Merge_Edge_Vectors(std::forward<const std::shared_ptr<Edge_t>>(edges),
      std::forward<const std::shared_ptr<uint32_t>>(edge_counts),
      std::forward<const std::shared_ptr<uint32_t>>(N_edges_tot), N_rngs, N_per_work_item));
    });
    return sort_event;
  }

  template <typename RNG>
  std::vector<Edge_t> random_connect(sycl::queue& q, const std::vector<uint32_t>& from,
                                     const std::vector<uint32_t>& to,
                                     const std::vector<uint32_t>& seeds, const auto p) {
    if (p == 1) {
      return complete_graph(from.size() + to.size(), false, true);
    } else if (p == 0) {
      return {};
    }

    std::vector<RNG> rngs;
    rngs.reserve(seeds.size());
    for (auto seed : seeds) {
      rngs.emplace_back(RNG(seed));
    }
    auto p_from = make_shared_usm(q, from);
    auto p_to = make_shared_usm(q, to);
    auto p_rng = make_shared_usm(q, rngs);
    auto N_edges_max = bipartite_graph_max_edges(from.size(), to.size());
    auto p_edges = make_shared_usm<Edge_t>(q, N_edges_max);
    auto p_N_edges_tot = make_shared_usm<uint32_t>(q, 1);

    auto event = random_connect<RNG>(q, p_from, p_to, p_rng, p, from.size(), to.size(), seeds.size(), N_edges_max, p_edges,
                                     p_N_edges_tot);
    event.wait();
    std::vector<Edge_t> edges(p_N_edges_tot.get()[0]);
    std::copy(p_edges.get(), p_edges.get() + p_N_edges_tot.get()[0], edges.begin());
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
