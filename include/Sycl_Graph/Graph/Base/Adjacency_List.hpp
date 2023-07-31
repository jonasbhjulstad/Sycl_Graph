#ifndef SYCL_GRAPH_GRAPH_BASE_ADJACENCY_LIST_HPP
#define SYCL_GRAPH_GRAPH_BASE_ADJACENCY_LIST_HPP
#include <Sycl_Graph/Buffer/Base/Edge_Buffer.hpp>
#include <cstdint>
#include <list>
#include <vector>

namespace Sycl_Graph {
  typedef std::vector<std::list<uint32_t>> Adjacency_List_t;

  Adjacency_List_t get_adjacency_list(const Edge_Buffer_type auto& edge_buf) {
    auto edges = edge_buf.get_edges();
    std::vector<std::list<uint32_t>> adj_list(edge_buf.N_vertices_max());
    for (auto& edge : edges) {
      adj_list[edge.first].push_back(edge.second);
    }
    return adj_list;
  }

  Adjacency_List_t merge_adjacency_lists(const Adjacency_List_t& adj_list_1,
                                         const Adjacency_List_t& adj_list_2) {
    Adjacency_List_t adj_list(adj_list_1.size() + adj_list_2.size());
    for (uint32_t i = 0; i < adj_list_1.size(); i++) {
      adj_list[i] = adj_list_1[i];
    }
    for (uint32_t i = 0; i < adj_list_2.size(); i++) {
      adj_list[i + adj_list_1.size()] = adj_list_2[i];
    }
    return adj_list;
  }

  Adjacency_List_t merge_adjacency_lists(const auto&&... adj_lists) {
    Adjacency_List_t adj_list(0);
    for (auto&& adj_list_i : {adj_lists...}) {
      adj_list = merge_adjacency_lists(adj_list, adj_list_i);
    }
    return adj_list;
  }

  Adjacency_List_t get_adjacency_list(const Graph_type auto& graph) {
    return get_adjacency_list(graph.edge_buf);
  }

}  // namespace Sycl_Graph

#endif
