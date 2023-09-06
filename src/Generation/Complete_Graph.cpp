#include <Sycl_Graph/Generation/Complete_Graph.hpp>
#include <execution>
#include <itertools.hpp>
namespace Sycl_Graph {
  std::vector<Edge_t> complete_graph(std::size_t N_vertices, bool directed, bool self_loops) {
    std::size_t N_edges = ((N_vertices * (N_vertices - 1) / 2) + (self_loops ? N_vertices : 0))
                          * (directed ? 2 : 1);
    std::vector<Edge_t> edges(N_edges);
    if (edges.max_size() < N_edges) {
      throw std::runtime_error("Too many edges requested");
    }
    std::vector<uint32_t> idx_0(N_vertices);
    std::iota(idx_0.begin(), idx_0.end(), 0);

    uint32_t i = 0;
    if (directed) {
      for (auto&& comb : iter::combinations_with_replacement(idx_0, 2)) {
        edges[2 * i] = Edge_t(comb[0], comb[1]);
        edges[2 * i + 1] = Edge_t(comb[1], comb[0]);
        i++;
      }
    } else {
      for (auto&& comb : iter::combinations_with_replacement(idx_0, 2)) {
        edges[i] = Edge_t(comb[0], comb[1]);
        i++;
      }
    }
    if (!self_loops) {
      edges.erase(
          std::remove_if(edges.begin(), edges.end(), [](Edge_t e) { return e.first == e.second; }),
          edges.end());
    }
    return edges;
  }
}  // namespace Sycl_Graph
