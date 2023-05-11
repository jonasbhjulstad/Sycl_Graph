#ifndef SYCL_GRAPH_ALGORITHMS_PROPERTIES_SYCL_PROPERTY_EXTRACTOR_HPP
#define SYCL_GRAPH_ALGORITHMS_PROPERTIES_SYCL_PROPERTY_EXTRACTOR_HPP

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

  namespace _detail {
    template <Graph_type Graph_t, Operation_type Op, std::size_t... Indices>
    auto get_graph_accessor_impl(Graph_t& graph, sycl::handler& h,
                                 const std::index_sequence<Indices...>&) {
      return std::make_tuple(
          graph.template get_access<std::get<Indices>(Op::graph_access_modes),
                                    std::tuple_element_t<Indices, typename Op::Accessor_Types>>(
              h)...);
    }
  }  // namespace _detail

  template <Graph_type Graph_t, Operation_type Op>
  auto get_graph_accessors(Graph_t& graph, sycl::handler& h) {
    // create index sequence for 'typename Op::Accessor_Types'
    constexpr size_t size = std::tuple_size<typename Op::Accessor_Types>::value;
    constexpr auto seq = std::make_integer_sequence<size_t, size>();
    return _detail::get_graph_accessor_impl<Graph_t, Op>(graph, h, seq);
  }

  template <Graph_type Graph_t, Operation_type Op>
  sycl::event invoke_transform(Graph_t& graph, const Op& operation, auto source_buf,
                               auto target_buf, sycl::event dep_event = {}) {
    return graph.q.submit({[&](sycl::handler& h) {
      auto source_acc = source_buf->template get_access<sycl::access_mode::read>(h);
      auto target_acc = target_buf->template get_access<Op::target_access_mode>(h);
      auto accessors = get_graph_accessors<Graph_t, Op>(graph, h);
      operation(accessors, source_acc, target_acc, h);
    }});
  }

  template <Graph_type Graph_t, Operation_type Op>
  sycl::event invoke_injection(Graph_t& graph, const Op& operation, auto source_buf,
                               sycl::event dep_event = {}) {
    return graph.q.submit({[&](sycl::handler& h) {
      auto source_acc = source_buf->template get_access<sycl::access_mode::read>(h);
      auto accessors = get_graph_accessors<Graph_t, Op>(graph, h);
      operation(accessors, source_acc, h);
    }});
  }

  template <Graph_type Graph_t, Operation_type Op>
  sycl::event invoke_extraction(Graph_t& graph, const Op& operation,
                                auto target_buf, sycl::event dep_event = {}) {
    return graph.q.submit([&](sycl::handler& h) {
      auto target_acc = target_buf->template get_access<Op::target_access_mode>(h);
      auto accessors = get_graph_accessors<Graph_t, Op>(graph, h);
      operation._invoke(accessors, target_acc, h);
    });
  }

  template <Graph_type Graph_t, Operation_type Op>
  sycl::event invoke_inplace_modification(Graph_t& graph, const Op& operation,
                                          sycl::event dep_event = {}) {
    return graph.q.submit({[&](sycl::handler& h) {
      auto accessors = get_graph_accessors<Graph_t, Op>(graph, h);
      operation(accessors, h);
    }});
  }

  template <Graph_type Graph_t, Operation_type Op>
  sycl::event invoke_operation(const Graph_t& G, const Op& operation, auto source_buf, auto target_buf)
  {
    if constexpr(has_Source_v<Op> && has_Target_v<Op>) {
      return invoke_transform(G, operation, source_buf, target_buf);
    } else if constexpr(has_Source_v<Op> && !has_Target_v<Op>) {
      return invoke_injection(G, operation, source_buf);
    } else if constexpr(!has_Source_v<Op> && has_Target_v<Op>) {
      return invoke_extraction(G, operation, target_buf);
    } else if constexpr(!has_Source_v<Op> && !has_Target_v<Op>) {
      return invoke_inplace_modification(G, operation);
    }
    return {};
  }

  template <Graph_type Graph_t, Operation_type... Op, tuple_like Source_Bufs_t, tuple_like Target_Bufs_t>
  auto invoke_operations(Graph_t& graph, const std::tuple<Op...>& operations,
                         Source_Bufs_t& source_bufs,
                         Target_Bufs_t& target_bufs) {

    auto test = std::tuple_cat(source_bufs, target_bufs);

    //assert that size of source_bufs and target_bufs is the same
    static_assert(std::tuple_size_v<Source_Bufs_t> == std::tuple_size_v<Target_Bufs_t>);

    multi_apply(invoke_operation<Graph_t, , operations, source_bufs, target_bufs);

    // return std::apply(
    //     [&](auto&... source_buf) {
    //       return std::apply(
    //           [&](auto&... target_buf) {
    //             return std::apply(
    //                 [&](const auto&&... op) {
    //                   return std::make_tuple(
    //                       invoke_operation(graph, op, source_buf, target_buf)...);
    //                 },
    //                 operations);
    //           },
    //           target_bufs);
    //     },
    //     source_bufs);

    
  }

  template <Graph_type Graph_t, Operation_type... Op>
  auto apply_single_operations(Graph_t& G, const std::tuple<Op...>& operations) {
    auto [source_bufs, target_bufs] = create_operation_buffers(G, operations);
    auto events = invoke_operations(G, operations, source_bufs, target_bufs);
    G.q.wait();
    return std::make_tuple(buffer_get(source_bufs), buffer_get(target_bufs));
  }

}  // namespace Sycl_Graph::Sycl
#endif