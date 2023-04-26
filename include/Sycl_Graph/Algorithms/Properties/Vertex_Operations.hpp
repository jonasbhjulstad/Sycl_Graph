#ifndef SYCL_GRAPH_ALGORITHMS_PROPERTIES_EDGE_OPERATIONS_HPP
#define SYCL_GRAPH_ALGORITHMS_PROPERTIES_EDGE_OPERATIONS_HPP
#include <Sycl_Graph/Algorithms/Properties/Operation_Types.hpp>
namespace Sycl_Graph::Sycl {

  template <Sycl_Graph::Sycl::Graph_type Graph_t, Operation_type Op>
  sycl::event vertex_extraction(Graph_t& graph, sycl::queue& q,
                                                 const Op& operation,
                                                 sycl::buffer<typename Op::Result_t>& result_buf) {
    using Vertex_t = typename Op::Vertex_t;
    return q.submit([&](sycl::handler& h) {
      auto result_acc = result_buf.template get_access<Op::result_access_mode>(h);
      auto vertex_acc = graph.template get_vertex_access<sycl::access::mode::read, Vertex_t>(h);
      operation(vertex_acc, result_acc, h);
    });
  }

  template <Sycl_Graph::Sycl::Graph_type Graph_t, Operation_type Op>
  sycl::event vertex_injection(Graph_t& graph, sycl::queue& q,
                                                 const Op& operation,
                                                 sycl::buffer<typename Op::Source_t>& source_buf) {
    using Vertex_t = typename Op::Vertex_t;
    return q.submit([&](sycl::handler& h) {
      auto source_acc = source_buf.template get_access<sycl::access::mode::read>(h);
      auto vertex_acc
          = graph.template get_vertex_access<sycl::access::mode::read_write, Vertex_t>(h);
      operation(source_acc, vertex_acc, h);
    });
  }
}  // namespace Sycl_Graph::Sycl
#endif