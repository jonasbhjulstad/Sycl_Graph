#ifndef SYCL_GRAPH_ALGORITHMS_PROPERTIES_OPERATION_BUFFERS_HPP
#define SYCL_GRAPH_ALGORITHMS_PROPERTIES_OPERATION_BUFFERS_HPP
#include <Sycl_Graph/Algorithms/Operations/Operation_Types.hpp>

namespace Sycl_Graph::Sycl {



  template <Graph_type Graph_t, Operation_type Op>
  sycl::buffer<typename Op::Target_t> create_result_buffer(const Graph_t& G, const Op& operation) {
    auto size = operation.target_buffer_size(G);
    auto buf = sycl::buffer<typename Op::Target_t>(sycl::range<1>(size));
    buffer_fill(buf, typename Op::Target_t(), G.q);
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
    auto size = operation.source_buffer_size(G);
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
    using Op_0_t = decltype(op_0);

    if constexpr (has_Source_t<Op_0_t> && has_Target_t<Op_0_t>) {
      auto source_bufs = create_source_buffers(G, operations);
      auto result_bufs = create_result_buffers(G, operations);
      return std::apply(
          [&](auto&&... first_buf) {
            return std::apply(
                [&](auto&&... second_buf) {
                  return std::make_tuple(std::make_pair(first_buf, second_buf)  ...);
                },
                result_bufs);
          },
          source_bufs);
    } else if constexpr (has_Source_t<Op_0_t>) {
      auto source_bufs = create_source_buffers(G, operations);
      return source_bufs;
    } else if constexpr (has_Target_t<Op_0_t>) {
      auto result_bufs = create_result_buffers(G, operations);
      return result_bufs;
    }
    static_assert(has_Source_t<Op_0_t> || has_Target_t<Op_0_t>,
                  "Operation must have either a source or target type");
  }

  template <Operation_type... Op>
  auto read_operation_buffers(std::tuple<sycl::buffer<typename Op::Target_t>...>& result_bufs) {
    return std::apply([&](auto&... buf) { return std::make_tuple(buffer_get(buf)...); },
                      result_bufs);
  }

  template <Operation_type... Op> using Operation_Buffer_Pairs_t = std::tuple<
      std::pair<sycl::buffer<typename Op::Source_t>, sycl::buffer<typename Op::Target_t>>...>;

  template <Operation_type... Op>
  auto read_operation_buffers(const Operation_Buffer_Pairs_t<Op...>& buffer_pairs) {
    return std::apply(
        [&](const auto&... buf) {
          return std::make_tuple(std::make_pair(buffer_get(buf.first), buffer_get(buf.second))...);
        },
        buffer_pairs);
  }


  template <Operation_type Op>
  auto create_operation_buffer_pair(const Op& operation)
  {
    if constexpr (is_Extraction_Operation_type<Op>)
    {
      return std::make_pair({}, create_result_buffer(G, operation));
    }
    else if constexpr(is_Injection_Operation_type<Op>)
    {
      return std::make_pair(create_source_buffer(G, operation), {});
    }
    else if constexpr(is_Transformation_Operation_type<Op>)
    {
      return std::make_pair(create_source_buffer(G, operation), create_result_buffer(G, operation));
    }
    static_assert(is_Extraction_Operation_type<Op> || is_Injection_Operation_type<Op> || is_Transformation_Operation_type<Op>,
                  "Operation must have either a source or target type");
  }


  template <Operation_type First, Operation_type Second, Operation_type... Op>
  struct Buffer_Sequence_Generator
  {
    auto _create_buffers(const std::tuple<First, Second, Op ...>& operations)
    {

    }

    auto create_buffers(const std::tuple<First, Second, Op ...>& operations)
    {
      auto [source_buf, target_buf] = create_operation_buffer_pair(std::get<0>(operations));
      return 
    }
  }

}  // namespace Sycl_Graph::Sycl
#endif