#include <Sycl_Graph/Metrics/Edge_Limits.hpp>
#include <itertools.hpp>
#include <numeric>
#include <Sycl_Graph/Utils/Math_Inl.hpp>
namespace Sycl_Graph {

  std::size_t complete_graph_max_edges(std::size_t N_vertices, bool directed, bool self_loops) {
    std::size_t N_edges
        = ((N_vertices * (N_vertices - 1))/2 + (self_loops ? N_vertices : 0))*(directed ? 2 : 1) ;
    return N_edges;
  }

  std::size_t complete_digraph_max_edges(std::size_t N_vertices, bool self_loops) {
    return complete_graph_max_edges(N_vertices, true, self_loops);
  }

  std::size_t bipartite_graph_max_edges(std::size_t N_vertices_0, std::size_t N_vertices_1,
                                        bool directed) {
    return std::pow(N_vertices_0 + N_vertices_1, 2) * (directed ? 2 : 1) / 4;
  }
  std::size_t SBM_expected_edges(const std::vector<uint32_t>& community_sizes,
                                 const std::vector<float>& p, bool directed,
                                 bool self_loops) {
    auto N_communities = community_sizes.size();
    std::vector<uint32_t> c_idx(N_communities);
    std::iota(c_idx.begin(), c_idx.end(), 0);
    auto N_edges_tot = 0;
    auto idx = 0;
    for (auto&& comb : iter::combinations_with_replacement(c_idx, 2)) {
      if (comb[0] == comb[1])
      {
        N_edges_tot += complete_graph_max_edges(community_sizes[comb[0]], directed, self_loops) * p[idx];
      }
      else
      {
        N_edges_tot += bipartite_graph_max_edges(community_sizes[comb[0]], community_sizes[comb[1]], directed)* p[idx];
      }
      idx++;
    }
    return N_edges_tot;
  }


  std::vector<std::size_t> SBM_max_connection_edges(const std::vector<uint32_t>& commmunity_sizes)
  {
    auto N_communities = commmunity_sizes.size();
    std::vector<uint32_t> c_idx(N_communities);
    auto N_connections = complete_graph_max_edges(N_communities, false, true);
    std::iota(c_idx.begin(), c_idx.end(), 0);
    std::vector<std::size_t> N_edges(N_connections);
    auto idx = 0;
    for (auto&& comb : iter::combinations_with_replacement(c_idx, 2)) {
      if (comb[0] == comb[1])
      {
        N_edges[idx] += complete_graph_max_edges(commmunity_sizes[comb[0]], false, true);
      }
      else
      {
        N_edges[idx] += bipartite_graph_max_edges(commmunity_sizes[comb[0]], commmunity_sizes[comb[1]], false);
      }
      idx++;
    }
    return N_edges;
  }


  std::vector<std::size_t> SBM_distributed_max_edges(const std::vector<uint32_t>& community_sizes, uint32_t N_max_edges)
  {

    auto max_edges = SBM_max_connection_edges(community_sizes);
    auto max_edges_tot = std::accumulate(max_edges.begin(), max_edges.end(), 0);
    if (N_max_edges > max_edges_tot)
    {
      return max_edges;
    }
    else
    {
      auto scale = static_cast<double>(N_max_edges) / max_edges_tot;
      std::vector<std::size_t> N_edges(max_edges.size());
      std::transform(max_edges.begin(), max_edges.end(), N_edges.begin(), [scale](std::size_t N) { return static_cast<std::size_t>(std::round(N * scale)); });
      return N_edges;
    }
  }



}  // namespace Sycl_Graph
