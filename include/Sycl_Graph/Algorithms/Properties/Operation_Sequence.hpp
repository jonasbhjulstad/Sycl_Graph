#ifndef SYCL_GRAPH_ALGORITHMS_PROPERTIES_OPERATION_PACK_HPP
#define SYCL_GRAPH_ALGORITHMS_PROPERTIES_OPERATION_PACK_HPP
#include <Sycl_Graph/Algorithms/Properties/Operation_Types.hpp>
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
  concept Operation_Sequence_type
   = tuple_like<T>;

  template <Operation_type ... Op>
  using Operation_Sequence_t = std::tuple<Op...>;

}
#endif