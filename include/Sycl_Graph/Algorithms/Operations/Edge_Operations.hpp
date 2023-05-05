#ifndef SYCL_GRAPH_ALGORITHMS_PROPERTIES_EDGE_OPERATIONS_HPP
#define SYCL_GRAPH_ALGORITHMS_PROPERTIES_EDGE_OPERATIONS_HPP
#include <Sycl_Graph/Algorithms/Operations/Operation_Types.hpp>
#include <Sycl_Graph/Graph/Sycl/Graph.hpp>
namespace Sycl_Graph::Sycl {
  template <Graph_type Graph_t, Operation_type Op>
  sycl::event edge_extraction(Graph_t& graph, const Op& operation,
                              sycl::buffer<typename Op::Target_t>& result_buf,
                              sycl::event dep_event = {}) {
    using Edge_t = typename Op::Edge_t;
    using From_t = typename Edge_t::From_t;
    using To_t = typename Edge_t::To_t;

    return graph.q.submit([&](sycl::handler& h) {
      h.depends_on(dep_event);
      auto result_acc = result_buf.template get_access<Op::target_access_mode>(h);
      auto edge_acc = graph.template get_edge_access<sycl::access::mode::read, Edge_t>(h);
      auto from_acc = graph.template get_vertex_access<sycl::access::mode::read, From_t>(h);
      auto to_acc = graph.template get_vertex_access<sycl::access::mode::read, To_t>(h);
      operation(edge_acc, from_acc, to_acc, result_acc, h);
    });
  }

  template <Graph_type Graph_t, Operation_type Op>
  sycl::event edge_injection(Graph_t& graph, const Op& operation,
                             sycl::buffer<typename Op::Source_t>& source_buf,
                             sycl::event dep_event = {}) {
    using Edge_t = typename Op::Edge_t;
    using From_t = typename Edge_t::From_t;
    using To_t = typename Edge_t::To_t;
    return graph.q.submit([&](sycl::handler& h) {
      h.depends_on(dep_event);
      auto source_acc = source_buf.template get_access<sycl::access::mode::read>(h);
      auto edge_acc = graph.template get_edge_access<Op::target_access_mode, Edge_t>(h);
      operation(source_acc, edge_acc, h);
    });
  }

  template <Graph_type Graph_t, Operation_type Op>
  sycl::event edge_transform(Graph_t& graph, const Op& operation,
                             sycl::buffer<typename Op::Source_t>& source_buf,
                             sycl::buffer<typename Op::Target_t>& target_buf,
                             sycl::event dep_event = {}) {
    using Edge_t = typename Op::Edge_t;
    using From_t = typename Edge_t::From_t;
    using To_t = typename Edge_t::To_t;
    return graph.q.submit([&](sycl::handler& h) {
      auto edge_acc = graph.template get_edge_access<sycl::access::mode::read, Edge_t>(h);
      auto from_acc = graph.template get_vertex_access<sycl::access::mode::read, From_t>(h);
      auto to_acc = graph.template get_vertex_access<sycl::access::mode::read, To_t>(h);
      auto source_acc = source_buf.template get_access<sycl::access::mode::read>(h);
      auto target_acc = target_buf.template get_access<Op::target_access_mode>(h);
      operation(edge_acc, from_acc, to_acc, source_acc, target_acc, h);
    });
  }

  template <Graph_type Graph_t, Operation_type Op>
  sycl::event edge_inplace_modification(Graph_t& graph, Op& operation, sycl::event dep_event = {}) {
    using Edge_t = typename Op::Edge_t;
    using From_t = typename Edge_t::From_t;
    using To_t = typename Edge_t::To_t;
    return graph.q.submit([&](sycl::handler& h) {
      auto edge_acc = graph.template get_edge_access<Op::inplace_access_mode, Edge_t>(h);
      auto from_acc = graph.template get_vertex_access<sycl::access::mode::read, From_t>(h);
      auto to_acc = graph.template get_vertex_access<sycl::access::mode::read, To_t>(h);
      operation(edge_acc, from_acc, to_acc, h);
    });
  }

}  // namespace Sycl_Graph::Sycl

#endif