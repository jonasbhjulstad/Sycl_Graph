#ifndef SYCL_GRAPH_ALGORITHMS_PROPERTIES_OPERATION_BUFFERS_HPP
#define SYCL_GRAPH_ALGORITHMS_PROPERTIES_OPERATION_BUFFERS_HPP
#include <Sycl_Graph/Algorithms/Operations/Operation_Types.hpp>

namespace Sycl_Graph::Sycl {
  template <Graph_type Graph_t, Operation_type Op>
  sycl::buffer<typename Op::Result_t> create_result_buffer(const Graph_t& G, const Op& operation) {
    auto size = operation.result_buffer_size(G);
    auto buf = sycl::buffer<typename Op::Result_t>(sycl::range<1>(size));
    buffer_fill(buf, typename Op::Result_t(), G.q);
    return buf;
  }

  template <Graph_type Graph_t, Operation_type... Op>
  auto create_result_buffers(const Graph_t& G, const std::tuple<Op...>& operations) {
    return std::apply(
        [&](const auto&... op) { return std::make_tuple(create_result_buffer(G, op)...); },
        operations);
  }

  template <Graph_type Graph_t, Operation_type Op>
  sycl::buffer<typename Op::Source_t> create_source_buffer(const Graph_t& G, const Op& operation) {
    auto size = operation.source_size(G);
    auto buf = sycl::buffer<typename Op::Source_t>(sycl::range<1>(size));
    buffer_fill(buf, typename Op::Source_t(), G.q);
  }

  template <Graph_type Graph_t, Operation_type... Op>
  auto create_source_buffers(const Graph_t& G, const std::tuple<Op...>& operations) {
    return std::apply(
        [&](const auto&... op) { return std::make_tuple(create_source_buffer(G, op)...); },
        operations);
  }

  template <Graph_type Graph_t, Operation_type... Op>
  auto create_operation_buffers(const Graph_t& G, const std::tuple<Op...>& operations) {
    auto op_0 = std::get<0>(operations);

    if constexpr (op_0.operation_type == Operation_Buffer_Transform) {
      auto source_bufs = create_source_buffers(G, operations);
      auto result_bufs = create_result_buffers(G, operations);
      return std::make_pair(source_bufs, result_bufs);
    } else if constexpr (op_0.operation_type == Operation_Direct_Transform) {
      auto result_bufs = create_result_buffers(G, operations);
      return result_bufs;
    }
  }

  template <Operation_type... Op>
  auto read_operation_buffers(std::tuple<sycl::buffer<typename Op::Result_t> ...>& result_bufs) {
    return std::apply([&](auto&... buf) { return std::make_tuple(buffer_get(buf)...); },
                      result_bufs);
  }

  template <Operation_type... Op> using Operation_Buffer_Pairs_t = std::tuple<
      std::pair<sycl::buffer<typename Op::Source_t>, sycl::buffer<typename Op::Result_t>>...>;

  template <Operation_type... Op>
  auto read_operation_buffers(const Operation_Buffer_Pairs_t<Op ...>& buffer_pairs) {
    return std::apply(
        [&](const auto&... buf) {
          return std::make_tuple(std::make_pair(buffer_get(buf.first), buffer_get(buf.second))...);
        },
        buffer_pairs);
  }

}  // namespace Sycl_Graph::Sycl
#endif