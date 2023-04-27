#ifndef SYCL_GRAPH_ALGORITHMS_PROPERTIES_SYCL_PROPERTY_EXTRACTOR_HPP
#define SYCL_GRAPH_ALGORITHMS_PROPERTIES_SYCL_PROPERTY_EXTRACTOR_HPP

#include <CL/sycl.hpp>
#include <Sycl_Graph/Algorithms/Operations/Edge_Operations.hpp>
#include <Sycl_Graph/Algorithms/Operations/Operation_Buffers.hpp>
#include <Sycl_Graph/Algorithms/Operations/Vertex_Operations.hpp>
#include <Sycl_Graph/Buffer/Sycl/type_helpers.hpp>
#include <Sycl_Graph/Graph/Sycl/Graph.hpp>
#include <tuple>

namespace Sycl_Graph::Sycl {

  template <Operation_type Op>
  sycl::event invoke_operation(sycl::queue& q, const Op& operation,
                               sycl::buffer<typename Op::Source_t>& source_buf,
                               sycl::buffer<typename Op::Result_t>& result_buf,
                               sycl::event dep_event = {}) {
    return q.submit([&](sycl::handler& h) {
      h.depends_on(dep_event);
      auto source_acc = source_buf.template get_access<sycl::access::mode::read>(h);
      auto result_acc = result_buf.template get_access<sycl::access::mode::read_write>(h);
      operation(source_acc, result_acc, h);
    });
  }

  template <Operation_type Op>
  sycl::event invoke_inplace_operation(sycl::queue& q, const Op& operation,
                                       sycl::buffer<typename Op::Result_t>& buf,
                                       sycl::event dep_event = {}) {
    return q.submit([&](sycl::handler& h) {
      h.depends_on(dep_event);
      auto buf_acc = result_buf.template get_access<sycl::access::mode::read_write>(h);
      operation(buf_acc, h);
    });
  }

  template <Sycl_Graph::Sycl::Graph_type Graph_t, Operation_type Op>
  sycl::event invoke_operation(Graph_t& G, sycl::queue& q, const Op& operation,
                               sycl::buffer<typename Op::Result_t>& result_buf,
                               sycl::event dep_event = {}) {
    if constexpr (Op::operation_target == Operation_Target_Vertex) {
      return vertex_extraction(G, q, operation, result_buf, dep_event);
    } else if constexpr (Op::operation_target == Operation_Target_Edge) {
      return edge_extraction(G, q, operation, result_buf, dep_event);
    } else {
      static_assert(operation.operation_target == Operation_Target_Vertex
                        || operation.operation_target == Operation_Target_Edge,
                    "Operation target must be either vertex or edge");
    }
  }

  template <Sycl_Graph::Sycl::Graph_type Graph_t, Operation_type Op>
  sycl::event invoke_operation(Graph_t&, sycl::queue& q, const Op& operation,
                               sycl::buffer<typename Op::Source_t>& source_buf,
                               sycl::event dep_event = {}) {
    if constexpr (operation.operation_target == Operation_Target_Vertex) {
      if constexpr (operation.operation_type == Operation_Modify_Vertices)
        return vertex_injection(q, operation, source_buf, dep_event);
    } else if constexpr (operation.operation_target == Operation_Target_Edge) {
      if constexpr (operation.operation_type == Operation_Modify_Edges)
        return edge_injection(q, operation, source_buf, dep_event);
    } else if constexpr (operation.operation_type == Operation_Modify_Inplace) {
      return invoke_inplace_operation(q, operation, source_buf, dep_event);
    } else {
      static_assert(operation.operation_target == Operation_Target_Vertex
                        || operation.operation_target == Operation_Target_Edge,
                    "Operation target must be either vertex or edge");
    }
  }

  template <Sycl_Graph::Sycl::Graph_type Graph_t, Operation_type Op>
  sycl::event invoke_operation(Graph_t&, sycl::queue& q, const Op& operation,
                               sycl::event dep_event = {}) {
    if constexpr (operation.operation_target == Operation_Target_Vertex
                  && operation.operation_type == Operation_Modify_Inplace)
      return vertex_modification(q, operation, dep_event);
    else if constexpr (operation.operation_target == Operation_Target_Edge
                       && operation.operation_type == Operation_Modify_Inplace)
      return edge_modification(q, operation, dep_event);
    else {
      static_assert(operation.operation_target == Operation_Target_Vertex
                        || operation.operation_target == Operation_Target_Edge,
                    "Operation target must be either vertex or edge");
    }
  }

  template <Graph_type Graph_t, Operation_type... Op>
  auto invoke_operations(Graph_t& graph, sycl::queue& q, const std::tuple<Op...>& operations,
                         std::tuple<sycl::buffer<typename Op::Result_t>...>& result_bufs) {
    return std::apply(
        [&](auto&... result_buf) {
          return std::apply(
              [&](const auto&... op) {
                return std::make_tuple(invoke_operation(graph, q, op, result_buf)...);
              },
              operations);
        },
        result_bufs);
  }

  template <Graph_type Graph_t, Operation_type... Op>
  auto invoke_operations(Graph_t& graph, sycl::queue& q, const std::tuple<Op...>& operations,
                         std::pair<std::tuple<sycl::buffer<typename Op::Source_t>...>,
                                   std::tuple<sycl::buffer<typename Op::Result_t>...>>& buffers,
                         auto& dep_events) {
    auto& source_bufs = buffers.first;
    auto& result_bufs = buffers.second;
    return std::apply(
        [&](auto&... dep_event) {
          return std::apply(
              [&](auto&... result_buf) {
                return std::apply(
                    [&](auto&... apply_buf) {
                      return std::apply(
                          [&](const auto&... op) {
                            return std::make_tuple(invoke_operation(graph, q, op, apply_buf,
                                                                    result_buf, dep_event)...);
                          },
                          operations);
                    },
                    source_bufs);
              },
              result_bufs);
        },
        dep_events);
  }

  template <Graph_type Graph_t, Operation_type... Op>
  auto apply_single_operations(Graph_t& G, const std::tuple<Op...>& operations) {
    auto bufs = create_operation_buffers(G, operations);
    auto events = invoke_operations(G, G.q, operations, bufs);
    G.q.wait();
    return read_operation_buffers<Op...>(bufs);
  }

}  // namespace Sycl_Graph::Sycl
#endif