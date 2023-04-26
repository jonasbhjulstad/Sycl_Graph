#ifndef SYCL_GRAPH_ALGORITHMS_PROPERTIES_SYCL_PROPERTY_EXTRACTOR_HPP
#define SYCL_GRAPH_ALGORITHMS_PROPERTIES_SYCL_PROPERTY_EXTRACTOR_HPP

#include <CL/sycl.hpp>
#include <Sycl_Graph/Algorithms/Properties/Vertex_Operations.hpp>
#include <Sycl_Graph/Algorithms/Properties/Edge_Operations.hpp>
#include <Sycl_Graph/Buffer/Sycl/type_helpers.hpp>
#include <Sycl_Graph/Graph/Sycl/Graph.hpp>
#include <tuple>

namespace Sycl_Graph::Sycl {

  template <Sycl_Graph::Sycl::Graph_type Graph_t, Operation_type Op>
  sycl::event invoke_operation(Graph_t&, const Op& operation, sycl::queue& q,
                                                sycl::buffer<typename Op::Source_t>& source_buf,
                                                sycl::buffer<typename Op::Result_t>& result_buf,
                                                sycl::event& dep_event) {
    return q.submit([&](sycl::handler& h) {
      h.depends_on(dep_event);
      auto source_acc = source_buf.template get_access<sycl::access::mode::read>(h);
      auto result_acc = result_buf.template get_access<Op::result_access_mode>(h);
      h.parallel_for(operation.N_parallel, [=](sycl::id<1> id) { operation(source_acc, result_acc, id); });
    });
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
  auto invoke_operations(Graph_t& graph, const std::tuple<Op...>& operations,
                         std::pair<std::tuple<sycl::buffer<typename Op::Source_t>...>,
                                   std::tuple<sycl::buffer<typename Op::Result_t>...>>& buffers,
                         sycl::queue& q, auto& dep_events) {
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
                            return std::make_tuple(invoke_operation(graph, op, apply_buf,
                                                                    result_buf, q, dep_event)...);
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
  std::tuple<std::vector<typename Op::Result_t>...> apply_operations(
      Graph_t& graph, const std::tuple<Op...>& operations, sycl::queue& q) {

    auto result_bufs = construct_buffers<typename Op::Result_t ..., Op::result_size ...>();

    auto& op_0 = std::get<0>(operations);
    if constexpr (op_0.operation_type == Operation_Buffer_Transform)
    {
      
      auto source_bufs = construct_buffers<typename Op::Source_t ...>();
      auto events = invoke_operations(graph, q, operations, std::make_pair(source_bufs, result_bufs));
    }
    else
    {
      auto events = invoke_operations(graph, q, operations, result_bufs);
    }

    q.wait();

    return buffer_get(result_bufs);
  }

}  // namespace Sycl_Graph::Sycl
#endif