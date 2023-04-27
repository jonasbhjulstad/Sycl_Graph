#ifndef SYCL_GRAPH_ALGORITHMS_PROPERTIES_SYCL_DEGREE_PROPERTIES_HPP
#define SYCL_GRAPH_ALGORITHMS_PROPERTIES_SYCL_DEGREE_PROPERTIES_HPP
#include <CL/sycl.hpp>
#include <Sycl_Graph/Algorithms/Operations/Operation_Types.hpp>
#include <Sycl_Graph/Buffer/Base/Vertex_Buffer.hpp>
#include <Sycl_Graph/Buffer/Base/Edge_Buffer.hpp>
#include <Sycl_Graph/Graph/Sycl/Graph.hpp>
#include <concepts>
namespace Sycl_Graph::Sycl {
  template <Sycl_Graph::Vertex_Buffer_type Vertex_Buffer_From_t,
            Sycl_Graph::Vertex_Buffer_type Vertex_Buffer_To_t,
            Sycl_Graph::Edge_Buffer_type Edge_Buffer_t>
  struct Directed_Vertex_Degree_Op {
    typedef uint32_t Result_t;
    enum Degree_Direction { Degree_Direction_From, Degree_Direction_To };

    static constexpr Operation_Target_t operation_target = Operation_Target_Edge;
    static constexpr Operation_Type_t operation_type = Operation_Direct_Transform;
    Degree_Direction direction = Degree_Direction_From;
    typedef typename Edge_Buffer_t::Edge_t Edge_t;
    typedef typename Vertex_Buffer_From_t::Vertex_t From_t;
    typedef typename Vertex_Buffer_To_t::Vertex_t To_t;

    Directed_Vertex_Degree_Op() = default;
    Directed_Vertex_Degree_Op(const std::tuple<Vertex_Buffer_From_t, const Vertex_Buffer_To_t,
                              const Edge_Buffer_t>&, Degree_Direction direction = Degree_Direction_From) {}

    void operator()(const auto& edge_acc, const auto& from_acc, const auto& to_acc,
                    auto& result_acc, sycl::handler& h) const {
      
      if (direction == Degree_Direction_From) {
        sycl::stream out(1024, 256, h);
        h.parallel_for(from_acc.size(), [=](sycl::id<1> id) {
          result_acc[id] = 0;
          for (int edge_idx = 0; edge_idx < edge_acc.size(); edge_idx++) {
            auto from_id = edge_acc[edge_idx].from;
            if(from_id == id) 
            result_acc[id] +=1;
          }
        });
      } else {
        h.parallel_for(to_acc.size(), [=](sycl::id<1> id) {
          result_acc[id] = 0;
          for (int edge_idx = 0; edge_idx < edge_acc.size(); edge_idx++) {
            auto to_id = edge_acc[edge_idx].to;
            if (to_id == id)
            result_acc[id] +=1;
          }
        });
      }
    }
    template <Graph_type Graph_t> size_t result_buffer_size(const Graph_t& G) const {
      if(direction == Degree_Direction_From)
        return G.vertex_buf.template get_buffer<Vertex_Buffer_From_t>().current_size();
      else
        return G.vertex_buf.template get_buffer<Vertex_Buffer_To_t>().current_size();
    }
  };



}  // namespace Sycl_Graph::Sycl
#endif