//
// Created by arch on 9/14/22.
//

#ifndef SYCL_GRAPH_GRAPH_GENERATION_HPP
#define SYCL_GRAPH_GRAPH_GENERATION_HPP
#include <Sycl_Graph/Graph/Sycl/Invariant_Graph.hpp>
#include <CL/sycl.hpp>
#include <itertools.hpp>
#include <memory>
#include <random>

namespace Sycl_Graph {
void random_connect(const std::vector<uint32_t>& from_nodes,
                           const std::vector<uint32_t> &to_nodes, float p,
                           bool self_loop, uint32_t seed) {
  uint32_t N_edges_max = 2 * to_nodes.size() * from_nodes.size();
  Edge_List_t edge_list;
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
    edge_list.erase(std::remove_if(edge_list.begin(), edge_list.end(),
                                   [&](auto &e) { return e.from == e.to; }),
                    edge_list.end());
  if (p == 1)
    return edge_list;
  if (p == 0)
    return Edge_List_t();
  edge_list.erase(std::remove_if(edge_list.begin(), edge_list.end(),
                                 [&](auto &e) { return !dist(rng); }),
                  edge_list.end());
  return edge_list;
}

void random_connect(Graph& G, float p, bool self_loop, uint32_t seed) {
  auto from_nodes = G.get_valid_nodes();
  auto to_nodes = G.get_valid_nodes();
  auto edge_list = random_connect(from_nodes, to_nodes, p, self_loop, seed);
  G.add_edges(edge_list);
}


} // namespace Sycl_Graph::Dynamic::Network_Models
#endif // FROLS_GRAPH_GENERATION_HPP
