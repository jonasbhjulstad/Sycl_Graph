#ifndef SYCL_GRAPH_ALGORITHMS_PROPERTIES_Operation_TYPES_HPP
#define SYCL_GRAPH_ALGORITHMS_PROPERTIES_Operation_TYPES_HPP
#include <CL/sycl.hpp>
#include <concepts>
#include <type_traits>
namespace Sycl_Graph::Sycl {
  enum Operation_Target_t { Operation_Target_Vertex, Operation_Target_Edge, Operation_Target_Buffer};

  enum Operation_Type_t {
    Operation_Direct_Transform,
    Operation_Buffer_Transform,
    Operation_Modify_Vertices,
    Operation_Modify_Edges,
    Operation_Modify_Inplace
  };

  template <typename T>
  concept Operation_type = requires(T t) {
                             T::operation_type;
                             T::operation_target;
                            //  T::operator();
                            //  T::result_buffer_size;
                           };

  template <typename T>
  concept Invariant_Operation_type = Operation_type<T>;


}  // namespace Sycl_Graph::Sycl

#endif