#ifndef SYCL_GRAPH_ALGORITHMS_PROPERTIES_OPERATION_BUFFERS_HPP
#define SYCL_GRAPH_ALGORITHMS_PROPERTIES_OPERATION_BUFFERS_HPP
#include <Sycl_Graph/Algorithms/Operations/Operation_Types.hpp>
#include <memory>
#include <type_traits>
namespace Sycl_Graph::Sycl {

  template <Graph_type Graph_t, typename Acc_t, Operation_type Op>
  auto find_create_buffer(const Graph_t& G, const Acc_t& accessor_type,
                          const Op& op) {
    using Data_t = typename Acc_t::Data_t;
    if constexpr (Graph_t::template has_type<Data_t>)
      return G.template get_buffer<Data_t>();
    else
      return std::make_shared<sycl::buffer<Data_t>>(
          sycl::buffer<Data_t>(sycl::range<1>(op.template _get_buffer_size<Data_t>(G))));
  }

  template <Graph_type Graph_t> auto find_create_buffers(const Graph_t& G, const auto& accessors) {
    return std::apply(
        [&](auto&&... accessor) { return std::make_tuple(find_create_buffer(G, accessor)...); },
        accessors);
  }

  template <Graph_type Graph_t, typename... Accessors_t>
  auto find_create_target_buffers(const Graph_t& G, const tuple_type auto& operations) {
    return std::apply(
        [&](const auto&&... op) {
          return std::make_tuple(find_create_buffers(G, op.target_accessors)...);
        },
        operations);
  }

  template <Graph_type Graph_t, Operation_type... Op>
  auto create_operation_buffer_sequence(const Graph_t& G, const std::tuple<Op...>& operations) {
    using Op_0 = std::tuple_element_t<0, std::tuple<Op...>>;
    const auto& op_0 = std::get<0>(operations);
    auto source_bufs_0 = find_create_buffers(G, Op_0::accessors);

    auto target_buffers = find_create_target_buffers(G, operations);
    auto source_buffers
        = std::tuple_cat(std::make_tuple(source_bufs_0), drop_last_tuple_elem(target_buffers));
    // verify_buffer_types(source_buffers, target_buffers);
    auto elem_test = std::get<0>(std::get<0>(source_buffers));
    return std::make_tuple(source_buffers, target_buffers);
  }

  // template <sycl::access::mode... Access_modes>
  // auto operation_buffer_access(sycl::handler& h, Graph_type auto& graph, tuple_type auto& bufs) {
  //   return std::apply(
  //       [&](auto&&... buf) {
  //         return std::make_tuple(buf->template get_access<Access_modes>(h, acc)...);
  //       },
  //       bufs);
  // }

  template <sycl::access::mode... Access_modes>
  auto operation_buffer_access(sycl::handler& h, const tuple_type auto& bufs) {
    return std::apply(
        [&](auto&&... buf) {
          return std::make_tuple(buf->template get_access<Access_modes>(h, buf)...);
        },
        bufs);
  }

}  // namespace Sycl_Graph::Sycl
#endif
