export module Operations.Degree_Properties;
#include <Sycl_Graph/Common/common.hpp>
import Operations.Types;
import Operations.Edge;
import Base.Buffer.Edge;
import Base.Buffer.Vertex;
import Sycl.Graph;
template <Edge_Buffer_type Edge_Buffer_t> export struct Directed_Vertex_Degree_Op
    : public Edge_Extract_Operation<Edge_Buffer_t, Directed_Vertex_Degree_Op<Edge_Buffer_t>>

{
  using Base_t = Edge_Extract_Operation<Edge_Buffer_t, Directed_Vertex_Degree_Op<Edge_Buffer_t>>;
  using Base_t::Base_t;
  typedef uint32_t Target_t;

  Directed_Vertex_Degree_Op() = default;
  Directed_Vertex_Degree_Op(const Edge_Buffer_t&) {}

  void invoke(auto& edge_acc, auto& from_acc, auto& to_acc, auto& target_acc,
              sycl::handler& h) const {
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
    auto to_size = G.template current_size<typename Base_t::To_t>();
    auto from_size = G.template current_size<typename Base_t::From_t>();
    return std::max(to_size, from_size);
  }
};
