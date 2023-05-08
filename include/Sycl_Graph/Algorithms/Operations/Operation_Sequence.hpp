#ifndef SYCL_GRAPH_ALGORITHMS_PROPERTIES_OPERATION_PACK_HPP
#define SYCL_GRAPH_ALGORITHMS_PROPERTIES_OPERATION_PACK_HPP
#include <Sycl_Graph/Algorithms/Operations/Operation_Types.hpp>
#include <Sycl_Graph/Algorithms/Operations/Operations.hpp>
#include <Sycl_Graph/type_helpers.hpp>
#include <vector>
namespace Sycl_Graph::Sycl {
  template <typename T>
  concept Operation_Pack_type
      = tuple_like<T>
        && requires(T t) {
             std::apply([&](auto&&... t) { requires(Operation_type<decltype(t)>); }, t);
           }

  template <typename T>
  concept Operation_Sequence_type = tuple_like<T>;

  template <Operation_type... Op> using Operation_Sequence_t = std::tuple<Op...>;

  template <Graph_type Graph_t, tuple_like Sequence_Bufs_t>
  auto op_sequence_invoke(Graph_t& G, const auto& op_sequence, Sequence_Bufs_t sequence_bufs,
                          sycl::event dep_event = {}) {
    if constexpr ((std::tuple_size_v<Source_Bufs_t> < 2)
                  || std::tuple_size_v<decltype(op_sequence)> == 0) {
      return sycl::event{};
    } else {
      dep_event = invoke_operation(G, q, std::get<0>(op_sequence), std::get<0>(sequence_bufs),
                                   std::get<1>(sequence_bufs), dep_event);
      auto buf_tail = drop_first_tuple_elem(sequence_bufs);
      return op_sequence_invoke(G, q, drop_first_tuple_elem(op_sequence), buf_tail, dep_event);
    }
  }

  template <Sycl_Graph::Sycl::Graph_type Graph_t, Operation_type Op_0, Operation_type... Op>
  auto invoke_operation_sequence(
      Graph_t&, sycl::queue& q, const std::tuple<Op_0, Op...>& op_sequence,
      std::tuple<sycl::buffer<typename Op_0::Source_t>, typename Op_0::Result_t,
                              typename Op::Result_t>...>& bufs,
      sycl::event dep_event = {}) {
    return op_sequence_invoke(G, bufs, dep_event);
  }

  template <Sycl_Graph::Sycl::Graph_type Graph_t, Operation_type... Op>
  sycl::event invoke_operation_sequence(
      Graph_t& G, sycl::queue& q, const std::tuple<Op...>& op_sequence,
      std::tuple<sycl::buffer<typename Op::Result_t>...>& result_buf, sycl::event dep_event = {}) {
    dep_event
        = invoke_operation(G, q, std::get<0>(op_sequence), std::get<0>(result_buf), dep_event);
    return op_sequence_invoke(G, q, drop_first_tuple_elem(op_sequence), result_buf, dep_event);
  }

  template <typename T>
  concept Operation_Sequence_type = tuple_like<T> && requires(T t) {
    std::apply([&](auto&&... t) { requires(Operation_type<decltype(t)>); }, t);

  };


}  // namespace Sycl_Graph::Sycl
#endif