//
// Created by arch on 9/14/22.
//

export module Base.Graph.Generation;
#include <Static_RNG/distributions.hpp>
#include <Sycl_Graph/Common/common.hpp>
#include <itertools.hpp>
import Base.Graph.Types;
import Base.Graph;

template <Edge_type Edge_t = ID_Edge_t>
export auto random_connect(const std::vector<uint32_t> &from_nodes,
                           const std::vector<uint32_t> &to_nodes, float p, bool self_loop,
                           uint32_t seed) {
  uint32_t N_edges_max = 2 * to_nodes.size() * from_nodes.size();
  std::vector<Edge_t> edge_list;
  edge_list.reserve(N_edges_max);
  std::random_device rd;
  std::vector<Static_RNG::default_rng> rngs;
  uint32_t n = 0;
  for (auto &&prod : iter::product(to_nodes, from_nodes)) {
    edge_list.push_back({std::get<0>(prod), std::get<1>(prod)});
    n++;
  }
  Static_RNG::default_rng rng(seed);
  Static_RNG::bernoulli_distribution<float> dist(p);
  if (!self_loop)
    edge_list.erase(
        std::remove_if(edge_list.begin(), edge_list.end(), [&](auto &e) { return e.from == e.to; }),
        edge_list.end());
  if (p == 1) return edge_list;
  if (p == 0) return std::vector<Edge_t>{};
  edge_list.erase(
      std::remove_if(edge_list.begin(), edge_list.end(), [&](auto &e) { return !dist(rng); }),
      edge_list.end());
  return edge_list;
}
