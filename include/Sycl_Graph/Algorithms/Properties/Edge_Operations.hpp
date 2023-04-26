#ifndef SYCL_GRAPH_ALGORITHMS_PROPERTIES_EDGE_OPERATIONS_HPP
#define SYCL_GRAPH_ALGORITHMS_PROPERTIES_EDGE_OPERATIONS_HPP
#include <Sycl_Graph/Algorithms/Properties/Extractor_Types.hpp>
namespace Sycl_Graph::Sycl {
  template <Sycl_Graph::Sycl::Graph_type Graph_t, Operation_type Op>
  sycl::event edge_extraction(Graph_t& graph, sycl::queue& q, const Op& operation,
                              sycl::buffer<typename Op::Result_t>& result_buf) {
    using Edge_t = typename Op::Edge_t;
    using From_t = typename Edge_t::From_t;
    using To_t = typename Edge_t::To_t;
    return q.submit([&](sycl::handler& h) {
      auto result_acc = result_buf.template get_access<Op::result_access_mode>(h);
      auto edge_acc = graph.template get_edge_access<sycl::access::mode::read, Edge_t>(h);
      auto from_acc = graph.template get_vertex_access<sycl::access::mode::read, From_t>(h);
      auto to_acc = graph.template get_vertex_access<sycl::access::mode::read, To_t>(h);
      operation(edge_acc, from_acc, to_acc, result_acc, h);
    });
  }

  template <Sycl_Graph::Sycl::Graph_type Graph_t, Operation_type Op>
  sycl::event edge_injection(Graph_t& graph, sycl::queue& q, const Op& operation,
                              sycl::buffer<typename Op::Source_t>& source_buf) {
    using Edge_t = typename Op::Edge_t;
    using From_t = typename Edge_t::From_t;
    using To_t = typename Edge_t::To_t;
    return q.submit([&](sycl::handler& h) {
      auto source_acc = source_buf.template get_access<sycl::access::mode::read>(h);
      auto edge_acc = graph.template get_edge_access<sycl::access::mode::read_write, Edge_t>(h);
      operation(source_acc, edge_acc, h);
    });
  }

}  // namespace Sycl_Graph::Sycl

#endif