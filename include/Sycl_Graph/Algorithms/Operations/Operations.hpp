#ifndef SYCL_GRAPH_ALGORITHMS_PROPERTIES_SYCL_PROPERTY_EXTRACTOR_HPP
#define SYCL_GRAPH_ALGORITHMS_PROPERTIES_SYCL_PROPERTY_EXTRACTOR_HPP

#include <CL/sycl.hpp>
#include <Sycl_Graph/Algorithms/Operations/Edge_Operations.hpp>
#include <Sycl_Graph/Algorithms/Operations/Operation_Buffers.hpp>
#include <Sycl_Graph/Algorithms/Operations/Operation_Types.hpp>
#include <Sycl_Graph/Algorithms/Operations/Vertex_Operations.hpp>
#include <Sycl_Graph/Buffer/Sycl/type_helpers.hpp>
#include <Sycl_Graph/Graph/Sycl/Graph.hpp>
#include <tuple>

namespace Sycl_Graph::Sycl {

  // template <Operation_type Op> using Buffer_Pair_t
  //     = std::tuple<sycl::buffer<typename Op::Source_t>, sycl::buffer<typename Op::Target_t>>;

  // template <Graph_type Graph_t, Operation_type Op>
  // sycl::event transform_dispatch(Graph_t& G, const Op& operation, Buffer_Pair_t<Op>& bufs,
  //                                sycl::event dep_event = {}) {
  //   if constexpr (has_Iterator_t<Op>) {
  //     if constexpr (Graph_t::template has_Edge_type<typename Op::Iterator_t>) {
  //       return edge_transform(G, operation, bufs.first, bufs.second, dep_event);
  //     } else if constexpr (Graph_t::template has_Vertex_type<typename Op::Iterator_t>) {
  //       return vertex_transform(G, operation, bufs.first, bufs.second, dep_event);
  //     }
  //   } else {
  //     return buffer_transform(G, operation, bufs.first, bufs.second, dep_event);
  //   }
  // }
  // template <Graph_type Graph_t, Operation_type Op>
  // sycl::event injection_dispatch(Graph_t& G, const Op& operation,
  //                                sycl::buffer<typename Op::Source_t>& buf,
  //                                sycl::event dep_event = {}) {
  //   if constexpr (Graph_t::template has_Edge_type<typename Op::Iterator_t>) {
  //     return edge_injection(G, operation, dep_event);
  //   } else if constexpr (Graph_t::template has_Vertex_type<typename Op::Iterator_t>) {
  //     return vertex_injection(G, operation, dep_event);
  //   }
  //   static_assert(Graph_t::template has_Edge_type<typename Op::Iterator_t>
  //                     || Graph_t::template has_Vertex_type<typename Op::Iterator_t>,
  //                 "Operation iterator must be either vertex or edge");
  // }
  // template <Graph_type Graph_t, Operation_type Op>
  // sycl::event extraction_dispatch(Graph_t& G, const Op& operation,
  //                                 sycl::buffer<typename Op::Target_t>& buf,
  //                                 sycl::event dep_event = {}) {
  //   if constexpr (Graph_t::template has_Edge_type<typename Op::Iterator_t>) {
  //     return edge_extraction(G, operation, buf, dep_event);
  //   } else if constexpr (Graph_t::template has_Vertex_type<typename Op::Iterator_t>) {
  //     return vertex_extraction(G, operation, buf, dep_event);
  //   }
  //   static_assert(Graph_t::template has_Edge_type<typename Op::Iterator_t>
  //                     || Graph_t::template has_Vertex_type<typename Op::Iterator_t>,
  //                 "Operation iterator must be either vertex or edge");
  // }

  // template <Graph_type Graph_t, Operation_type Op>
  // sycl::event inplace_dispatch(Graph_t& G, const Op& operation,
  //                              sycl::buffer<typename Op::Target_t>& buf,
  //                              sycl::event dep_event = {}) {
  //   if constexpr (has_Iterator_t<Op>) {
  //     if constexpr (Graph_t::template has_Edge_type<typename Op::Iterator_t>) {
  //       return edge_inplace_modification(G, operation, buf, dep_event);
  //     } else if constexpr (Graph_t::template has_Vertex_type<typename Op::Iterator_t>) {
  //       return vertex_inplace_modification(G, operation, buf, dep_event);
  //     }
  //     static_assert(Graph_t::template has_Edge_type<typename Op::Iterator_t>
  //                       || Graph_t::template has_Vertex_type<typename Op::Iterator_t>,
  //                   "Operation iterator must be either vertex or edge");
  //   } else {
  //     return buffer_inplace_modification(G, operation, buf, dep_event);
  //   }
  // }

  // template <Graph_type Graph_t, Operation_type Op>
  // sycl::event invoke_operation(Graph_t& G, const Op& operation, auto& bufs,
  //                              sycl::event dep_event = {}) {
  //   if constexpr (has_Inplace_t<Op>) {
  //     return inplace_dispatch(G, operation, bufs, dep_event);
  //   }
  //   if constexpr (has_Source_t<Op>) {
  //     if constexpr (has_Target_t<Op>) {
  //       return transform_dispatch(G, operation, bufs, dep_event);
  //     } else {
  //       return injection_dispatch(G, operation, bufs, dep_event);
  //     }
  //   } else if constexpr (has_Target_t<Op> && has_Iterator_t<Op>) {
  //     return extraction_dispatch(G, operation, bufs, dep_event);
  //   }
  //   static_assert(has_Source_t<Op> || has_Target_t<Op> || has_Iterator_t<Op>,
  //                 "Operation must have either source, target or iterator");
  // }

  template <Graph_type Graph_t, Operation_type Op>
  auto get_graph_accessors(Graph_t& graph, sycl::handler& h) {
    auto dummy = typename Op::Accessor_Types();
    auto accessors = std::apply(
        [&](auto&&... modes) {
          return std::apply(
              [&](auto&&... args) {
                return std::make_tuple(graph.template get_access<modes, decltype(args)>(h)...);
              },
              dummy);
        },
        Op::graph_access_modes);
  }

  template <Graph_type Graph_t, Operation_type Op>
  sycl::event invoke_operation(Graph_t& graph, Op& operation,
                               sycl::buffer<typename Op::Source_t>& source_buf,
                               sycl::buffer<typename Op::Target_t>& target_buf,
                               sycl::event dep_event = {}) {
    return graph.q.submit({[&](sycl::handler& h) {
      auto source_acc = source_buf.template get_access<sycl::access_mode::read>(h);
      auto target_acc = target_buf.template get_access<Op::target_access_mode>(h);
      auto accessors = get_graph_accessors<Graph_t, Op>(graph, h);
      operation(accessors, source_acc, target_acc, h);
    }});
  }

  template <Graph_type Graph_t, Operation_type Op>
  sycl::event invoke_operation(Graph_t& graph, Op& operation,
                               sycl::buffer<typename Op::Source_t>& source_buf,
                               sycl::event dep_event = {}) {
    return graph.q.submit({[&](sycl::handler& h) {
      auto source_acc = source_buf.template get_access<sycl::access_mode::read>(h);
      auto accessors = get_graph_accessors<Graph_t, Op>(graph, h);
      operation(accessors, source_acc, h);
    }});
  }

  template <Graph_type Graph_t, Operation_type Op>
  sycl::event invoke_operation(Graph_t& graph, Op& operation,
                               sycl::buffer<typename Op::Target_t>& target_buf,
                               sycl::event dep_event = {}) {
    return graph.q.submit({[&](sycl::handler& h) {
      auto target_acc = target_buf.template get_access<Op::target_access_mode>(h);
      auto accessors = get_graph_accessors<Graph_t, Op>(graph, h);
      operation(accessors, target_acc, h);
    }});
  }

  template <Graph_type Graph_t, Operation_type Op>
  sycl::event invoke_operation(Graph_t& graph, Op& operation, sycl::event dep_event = {}) {
    return graph.q.submit({[&](sycl::handler& h) {
      auto accessors = get_graph_accessors<Graph_t, Op>(graph, h);
      operation(accessors, h);
    }});
  }

  template <Graph_type Graph_t, Operation_type... Op, typename... Bufs_t>
  auto invoke_operations(Graph_t& graph, const std::tuple<Op...>& operations,
                         std::tuple<Bufs_t...>& bufs) {
    return std::apply(
        [&](auto&... target_buf) {
          return std::apply(
              [&](const auto&... op) {
                return std::make_tuple(invoke_operation(graph, op, target_buf)...);
              },
              operations);
        },
        bufs);
  }

  template <Graph_type Graph_t, Operation_type... Op>
  auto apply_single_operations(Graph_t& G, const std::tuple<Op...>& operations) {
    auto bufs = create_operation_buffers(G, operations);
    auto events = invoke_operations(G, operations, bufs);
    G.q.wait();
    return read_operation_buffers<Op...>(bufs);
  }

}  // namespace Sycl_Graph::Sycl
#endif