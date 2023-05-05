#ifndef SYCL_GRAPH_ALGORITHMS_PROPERTIES_SYCL_DEGREE_PROPERTIES_HPP
#define SYCL_GRAPH_ALGORITHMS_PROPERTIES_SYCL_DEGREE_PROPERTIES_HPP
#include <CL/sycl.hpp>
#include <Sycl_Graph/Algorithms/Operations/Operation_Types.hpp>
#include <Sycl_Graph/Buffer/Base/Edge_Buffer.hpp>
#include <Sycl_Graph/Buffer/Base/Vertex_Buffer.hpp>
#include <Sycl_Graph/Graph/Sycl/Graph.hpp>
#include <concepts>
namespace Sycl_Graph::Sycl {
  template <Sycl_Graph::Edge_Buffer_type Edge_Buffer_t> struct Directed_Vertex_Degree_Op {
    typedef typename Edge_Buffer_t::Edge_t Edge_t;
    typedef typename Edge_t::From_t From_t;
    typedef typename Edge_t::To_t To_t;

    typedef std::tuple<Edge_t, From_t, To_t> Accessor_Types;
    static constexpr sycl::access::mode graph_access_modes[]
        = {sycl::access_mode::read, sycl::access_mode::read, sycl::access_mode::read};
    typedef uint32_t Target_t;
    static constexpr sycl::access::mode target_access_mode = sycl::access_mode::write;

    Directed_Vertex_Degree_Op() = default;
    Directed_Vertex_Degree_Op(const Edge_Buffer_t&) {}

    void operator()(auto& accessors, auto& target_acc, sycl::handler& h) const {
      auto& from_acc = std::get<0>(accessors);
      auto& to_acc = std::get<1>(accessors);
      auto& edge_acc = std::get<2>(accessors);
      h.parallel_for(from_acc.size(), [=](sycl::id<1> id) {
        target_acc[id] = 0;
        for (int edge_idx = 0; edge_idx < edge_acc.size(); edge_idx++) {
          auto from_id = edge_acc[edge_idx].from;
          if (from_id == id) target_acc[id] += 1;

          auto to_id = edge_acc[edge_idx].to;
          if (to_id == id) target_acc[id] += 1;
        }
      });
    }
    template <Graph_type Graph_t> size_t target_buffer_size(const Graph_t& G) const {
      

      auto to_size = G.template current_size<To_t>();
      auto from_size = G.template current_size<From_t>();
      return std::max(to_size, from_size);
    }
  };

}  // namespace Sycl_Graph::Sycl
#endif