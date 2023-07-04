#ifndef SYCL_GRAPH_ALGORITHMS_PROPERTIES_OPERATION_BUFFERS_HPP
#define SYCL_GRAPH_ALGORITHMS_PROPERTIES_OPERATION_BUFFERS_HPP
#include <Sycl_Graph/Algorithms/Operations/Operation_Types.hpp>
#include <memory>
#include <type_traits>
namespace Sycl_Graph::Sycl {

  template <Graph_type Graph_t, Operation_type Op>
  auto create_source_buffers(const Graph_t& G, const Op op) {
    auto source_sizes = op.get_source_sizes(G);

    return std::apply(
        [&](auto&&... accessor_types) {
          return std::apply(
              [&](auto&&... source_size) {
                return std::make_tuple(
                    std::make_shared<sycl::buffer<
                        typename std::remove_reference_t<decltype(accessor_types)>::type>>(
                        sycl::buffer<
                            typename std::remove_reference_t<decltype(accessor_types)>::type>(
                            sycl::range<1>(source_size)))...);
              },
              source_sizes);
        },
        typename Op::Source_Accessors_t{});
  }

  template <Graph_type Graph_t, Operation_type Op>
  auto create_target_buffers(const Graph_t& G, const Op& op) {
    return std::apply(
        [&](auto&&... accessor_types) {
          return std::make_tuple(
              std::make_shared<
                  sycl::buffer<typename std::remove_reference_t<decltype(accessor_types)>::type>>(
                  sycl::buffer<typename std::remove_reference_t<decltype(accessor_types)>::type>(
                      sycl::range<1>(op.template _get_buffer_size<typename std::remove_reference_t<
                                         decltype(accessor_types)>::type>(G))))...);
        },
        typename Op::Target_Accessors_t{});
  }

  template <Graph_type Graph_t, Operation_type... Op>
  auto create_operation_buffer_sequence(const Graph_t& G, const std::tuple<Op...>& operations) {
    auto source_bufs_0 = create_source_buffers(G, std::get<0>(operations));
    auto target_buffers = std::apply(
        [&](auto&&... op) { return std::make_tuple(create_target_buffers(G, op)...); }, operations);
    auto source_buffers
        = std::tuple_cat(std::make_tuple(source_bufs_0), drop_last_tuple_elem(target_buffers));

    // verify_buffer_types(source_buffers, target_buffers);
    auto elem_test = std::get<0>(std::get<0>(source_buffers));
    return std::make_tuple(source_buffers, target_buffers);
  }



}  // namespace Sycl_Graph::Sycl
#endif
