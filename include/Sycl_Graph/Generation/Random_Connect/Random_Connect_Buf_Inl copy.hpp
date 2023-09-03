#ifndef SYCL_GRAPH_GENERATION_RANDOM_CONNECT_INL_HPP
#define SYCL_GRAPH_GENERATION_RANDOM_CONNECT_INL_HPP
#include <Sycl_Graph/Graph/Graph.hpp>
#include <Sycl_Graph/Metrics/Edge_Limits.hpp>
#include <Sycl_Graph/Utils/Buffer_Utils.hpp>
#include <Sycl_Graph/Utils/Math.hpp>
#include <Sycl_Graph/Utils/work_groups.hpp>
#include <oneapi/dpl/random>

namespace Sycl_Graph {

template <typename RNG>
  struct Random_Connect_Kernel {
    Random_Connect_Kernel(read_accessor<uint32_t>& from_acc,
                          read_accessor<uint32_t>& to_acc,
                          sycl::accessor<RNG>& rng_acc,
                          write_accessor<Edge_t>& e_acc,
                          write_accessor<uint32_t>& count_acc,
                          float p, uint32_t Npw)
        : from_acc(from_acc),
          to_acc(to_acc),
          rng_acc(rng_acc),
          e_acc(e_acc),
          count_acc(count_acc),
          N_from(from_acc.size()),
          N_to(to_acc.size()), p(p), N_per_work_item(Npw) {}

    void operator()(sycl::nd_item<1> it) const {
      auto gid = it.get_global_id();
      auto &rng = rng_acc[gid];
      oneapi::dpl::bernoulli_distribution dist(p);
      auto count = 0;
      for (auto e_idx = gid * N_per_work_item; e_idx < (gid + 1) * N_per_work_item; ++e_idx) {
        if (dist(rng)) {
          auto from_idx = floor_div(e_idx, N_from);
          auto to_idx = e_idx % N_to;
          e_acc[e_idx] = std::make_pair(from_idx, to_idx);
          count++;
        }
      }
      count_acc[gid] = count;
    }
    const uint32_t N_from, N_to, N_per_work_item;
    const float p;

  private:
    read_accessor<uint32_t> from_acc;
    read_accessor<uint32_t> to_acc;
    sycl::accessor<RNG> rng_acc;
    write_accessor<Edge_t> e_acc;
    write_accessor<uint32_t> count_acc;
  };


  struct Merge_Edge_Vectors {
    Merge_Edge_Vectors(sycl::accessor<Edge_t> e_acc,
                        read_accessor<uint32_t>& count_acc,
                       write_accessor<uint32_t>& Ntot_acc, uint32_t Npw)
        : e_acc(e_acc), count_acc(count_acc), Ntot_acc(Ntot_acc), N_per_work_item(Npw) {}
    void operator()() const {
      auto N_merged_edges = 0;
      for (int i = 0; i < count_acc.size(); i++) {
        for (int j = 0; j < count_acc[i]; j++) {
          e_acc[N_merged_edges] = e_acc[i * N_per_work_item + j];
          N_merged_edges++;
        }
      }
      Ntot_acc[0] = N_merged_edges;
    }

    const uint32_t N_per_work_item;

  private:
    sycl::accessor<Edge_t> e_acc;
    read_accessor<uint32_t> count_acc;
    write_accessor<uint32_t> Ntot_acc;
  };

  template <typename RNG>
  sycl::event random_connect(sycl::queue &q, sycl::buffer<uint32_t, 1> &from,
                             sycl::buffer<uint32_t, 1> &to, sycl::buffer<RNG> &rngs, const float p,
                             Edgebuf_t<1> &edges, sycl::buffer<uint32_t> &N_edges_tot) {
    auto N_edges_max = edges.get_range()[0];
    if (N_edges_max < bipartite_graph_max_edges(from.size(), to.size()) * p) {
      throw std::runtime_error(
          "Space allocated in edge buffer is less than the expected number of edges.");
    }
    auto N_rngs = rngs.get_range()[0];
    auto nd_range = get_nd_range(q, N_rngs);
    auto N_per_work_item = get_N_per_work_item(N_edges_max, nd_range);
    sycl::buffer<uint32_t> edge_counts((sycl::range<1>(N_rngs)));
    auto sample_event = q.submit([&](sycl::handler &h) {
      auto from_acc = from.template get_access<sycl::access::mode::read>(h);
      auto to_acc = to.template get_access<sycl::access::mode::read>(h);
      auto e_acc = edges.template get_access<sycl::access::mode::write>(h);
      auto rng_acc = rngs.template get_access<sycl::access::mode::read_write>(h);
      auto count_acc = edge_counts.template get_access<sycl::access::mode::write>(h);
      Random_Connect_Kernel<RNG> kernel(from_acc, to_acc, rng_acc, e_acc, count_acc, p, N_per_work_item);
      h.parallel_for(nd_range, kernel);
    });
    auto sort_event = q.submit([&](sycl::handler &h) {
      h.depends_on(sample_event);
      auto e_acc = edges.template get_access<sycl::access::mode::read_write>(h);
      auto count_acc = edge_counts.template get_access<sycl::access::mode::read>(h);
      auto Ntot_acc = N_edges_tot.template get_access<sycl::access::mode::write>(h);
      h.single_task(Merge_Edge_Vectors(e_acc, count_acc, Ntot_acc, N_per_work_item));
    });
    return sort_event;
  }

  template <typename RNG>
  std::vector<Edge_t> random_connect(sycl::queue &q, const std::vector<uint32_t> &from,
                                     const std::vector<uint32_t> &to,
                                     const std::vector<uint32_t> &seeds, const auto p) {
    std::vector<RNG> rngs;
    rngs.reserve(seeds.size());
    for (auto seed : seeds) {
      rngs.emplace_back(RNG(seed));
    }
    sycl::buffer<uint32_t, 1> from_buf(from.data(), sycl::range<1>(from.size()));
    sycl::buffer<uint32_t, 1> to_buf(to.data(), sycl::range<1>(to.size()));
    sycl::buffer<RNG, 1> rng_buf(rngs.data(), sycl::range<1>(rngs.size()));
    Edgebuf_t<1> edges_buf(sycl::range<1>(bipartite_graph_max_edges(from.size(), to.size())));
    sycl::buffer<uint32_t> N_edges_tot_buf(sycl::range<1>(1));

    auto event = random_connect<RNG>(q, from_buf, to_buf, rng_buf, p, edges_buf, N_edges_tot_buf);
    event.wait();
    std::vector<uint32_t> edge_counts(1);
    read_buffer(q, N_edges_tot_buf, edge_counts).wait();
    std::vector<Edge_t> edges(edge_counts[0]);
    read_buffer(q, edges_buf, edges).wait();
    return edges;
  }

  template <typename RNG> std::vector<Edge_t> self_connect(sycl::queue &q,
                                                           const std::vector<uint32_t> ids,
                                                           const std::vector<uint32_t> &seeds,
                                                           const auto p, bool self_loops) {
    auto edgelist = random_connect<RNG>(q, ids, ids, seeds, p);
    if (!self_loops) {
      edgelist.erase(std::remove_if(edgelist.begin(), edgelist.end(),
                                    [](auto e) { return e.first == e.second; }),
                     edgelist.end());
    }
    return edgelist;
  }

}  // namespace Sycl_Graph

#endif
