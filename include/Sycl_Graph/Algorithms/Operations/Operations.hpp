#ifndef SYCL_GRAPH_ALGORITHMS_PROPERTIES_SYCL_PROPERTY_EXTRACTOR_HPP
#define SYCL_GRAPH_ALGORITHMS_PROPERTIES_SYCL_PROPERTY_EXTRACTOR_HPP

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

#include <CL/sycl.hpp>
#include <Sycl_Graph/Algorithms/Operations/Edge_Operations.hpp>
#include <Sycl_Graph/Algorithms/Operations/Operation_Buffers.hpp>
#include <Sycl_Graph/Algorithms/Operations/Operation_Types.hpp>
#include <Sycl_Graph/Algorithms/Operations/Vertex_Operations.hpp>
#include <Sycl_Graph/Buffer/Sycl/type_helpers.hpp>
#include <Sycl_Graph/Graph/Sycl/Graph.hpp>
#include <array>
#include <tuple>

namespace Sycl_Graph::Sycl {

  template <Operation_type Op>
  sycl::event invoke_operation(Graph_type auto &graph, Op &operation,
                               const tuple_like auto &&source_bufs, tuple_like auto &&target_bufs,
                               tuple_like auto &custom_bufs, auto &dep_event) {
    return graph.q.submit([&](sycl::handler &h) {
      h.depends_on(dep_event);
      operation.__invoke(h, source_bufs, target_bufs, custom_bufs);
    });
  }

  sycl::event invoke_operation(Graph_type auto &G, tuple_like auto &&tuple) {
    return std::apply([&](auto &&...tup) { return invoke_operation(G, tup...); });
  }

  template <Operation_type... Op>
  auto invoke_operations(Graph_type auto &graph, std::tuple<Op...> &operations,
                         tuple_like auto &source_bufs, tuple_like auto &target_bufs,
                         tuple_like auto &custom_bufs,
                         UniformTuple<sizeof...(Op), sycl::event> dep_events
                         = UniformTuple<sizeof...(Op), sycl::event>{}) {
    static_assert(!std::is_same_v<decltype(std::get<0>(target_bufs)), std::nullptr_t>);

    auto shuffled_tuples
        = shuffle_tuples(operations, source_bufs, target_bufs, custom_bufs, dep_events);
    std::apply([&](auto &&...tup) { return std::make_tuple(invoke_operation(graph, tup)...); },
               shuffled_tuples);
  }
  template <Operation_type... Op>
  auto invoke_operations(Graph_type auto &graph, std::tuple<Op...> &operations,
                         tuple_like auto &source_bufs, tuple_like auto &target_bufs,
                         UniformTuple<sizeof...(Op), sycl::event> dep_events
                         = UniformTuple<sizeof...(Op), sycl::event>{}) {
    static_assert(!std::is_same_v<decltype(std::get<0>(target_bufs)), std::nullptr_t>);

    auto shuffled_tuples = shuffle_tuples(operations, source_bufs, target_bufs,
                                          EmptyTuple<sizeof...(Op)>{}, dep_events);
    std::apply([&](auto &&...tup) { return std::make_tuple(invoke_operation(graph, tup)...); },
               shuffled_tuples);
  }

  template <Operation_type... Op>
  auto invoke_operation_sequence(Graph_type auto &graph, std::tuple<Op...> &operations,
                                 tuple_like auto &source_bufs, tuple_like auto &target_bufs,
                                 tuple_like auto &custom_bufs,
                                 sycl::event dep_event = sycl::event{});

  template <Operation_type... Op>
  auto invoke_operation_sequence(Graph_type auto &graph, std::tuple<Op...> &operations,
                                 tuple_like auto &source_bufs, tuple_like auto &target_bufs,
                                 tuple_like auto &custom_bufs, sycl::event dep_event) {
    auto event = invoke_operation(graph, std::get<0>(operations), std::get<0>(source_bufs),
                                  std::get<0>(target_bufs), std::get<0>(custom_bufs), dep_event);
    if constexpr (std::tuple_size_v<std::tuple<Op...>> > 1) {
      auto other_events
          = invoke_operation_sequence(graph, tuple_tail(operations), tuple_tail(source_bufs),
                                      tuple_tail(target_bufs), tuple_tail(custom_bufs), event);
      return std::tuple_cat(std::make_tuple(event), other_events);
    } else {
      return std::make_tuple(event);
    }
  }

  template <Operation_type... Op>
  auto invoke_operation_sequence(Graph_type auto &graph, std::tuple<Op...> &&operations,
                                 tuple_like auto &&source_bufs, tuple_like auto &&target_bufs,
                                 tuple_like auto &&custom_bufs, sycl::event dep_event) {
    auto event = invoke_operation(graph, std::get<0>(operations), std::get<0>(source_bufs),
                                  std::get<0>(target_bufs), std::get<0>(custom_bufs), dep_event);
#ifdef OPERATION_DEBUG_TARGET_BUFS
    auto target_vec = buffer_get(std::get<0>(target_bufs));
#endif
    int a = 0;
    if constexpr (std::tuple_size_v<std::tuple<Op...>> > 1) {
      auto other_events
          = invoke_operation_sequence(graph, tuple_tail(operations), tuple_tail(source_bufs),
                                      tuple_tail(target_bufs), tuple_tail(custom_bufs), event);
      return std::tuple_cat(std::make_tuple(event), other_events);
    } else {
      return std::make_tuple(event);
    }
  }

}  // namespace Sycl_Graph::Sycl
#endif
