#include <Sycl_Graph/Generation/SBM/SBM_Types.hpp>
#include <Sycl_Graph/Utils/Buffer_Utils.hpp>
namespace Sycl_Graph {

  std::vector<Edge_t> SBM_Graph_t::get_flat_edges() const {
    std::vector<Edge_t> flat_edges;
    for (auto&& e : edges) {
      flat_edges.insert(flat_edges.end(), e.begin(), e.end());
    }
    return flat_edges;
  }
  std::vector<uint32_t> SBM_Graph_t::get_flat_vertices() const {
    std::vector<uint32_t> flat_vertices;
    for (auto&& v : vertices) {
      flat_vertices.insert(flat_vertices.end(), v.begin(), v.end());
    }
    return flat_vertices;
  }
  std::size_t SBM_Graph_t::get_N_edges() const {
    return std::accumulate(
        edges.begin(), edges.end(), 0,
        [](const uint32_t sum, const std::vector<Edge_t>& e) { return sum + e.size(); });
  }
  std::size_t SBM_Graph_t::get_N_vertices() const {
    return std::accumulate(
        vertices.begin(), vertices.end(), 0,
        [](const uint32_t sum, const std::vector<uint32_t>& v) { return sum + v.size(); });
  }

  std::size_t SBM_Graph_t::get_N_communities() const { return vertices.size(); }

  std::size_t SBM_Graph_t::get_N_connections() const { return edges.size(); }

    bool SBM_Graph_t::is_edges_valid() const
    {
      return std::all_of(edges.begin(), edges.end(), [](const auto& e_list)
      {
        return std::all_of(e_list.begin(), e_list.end(), [](auto e)
        {
          return e.is_valid();
        });
      });
    }

    bool SBM_Graph_t::is_vertices_valid() const
    {
      return std::all_of(vertices.begin(), vertices.end(), [](const auto& v_list)
      {
        return std::all_of(v_list.begin(), v_list.end(), [](auto v)
        {
          return v != std::numeric_limits<uint32_t>::max();
        });
      });
    }
    bool SBM_Graph_t::is_valid() const{
      return is_edges_valid() && is_vertices_valid();
    }


}  // namespace Sycl_Graph
