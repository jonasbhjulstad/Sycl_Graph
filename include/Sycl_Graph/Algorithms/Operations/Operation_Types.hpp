#ifndef SYCL_GRAPH_ALGORITHMS_PROPERTIES_Operation_TYPES_HPP
#define SYCL_GRAPH_ALGORITHMS_PROPERTIES_Operation_TYPES_HPP
#include <CL/sycl.hpp>
#include <concepts>
#include <type_traits>
#include <Sycl_Graph/Graph/Base/Graph_Types.hpp>
namespace Sycl_Graph::Sycl {


  template <typename T> 
  concept has_Graph_Iterator = true;

  template <typename T>
  concept Operation_type = true;

  template <typename T>
  concept _has_Source_t = requires(T t) {
    typename T::Source_t;
  };

  template <typename T>
  static constexpr bool has_Source_t = _has_Source_t<T>;

  template <typename T>
  concept _has_Target_t = requires(T t) {
    typename T::Target_t;
  };

  template <typename T>
  static constexpr bool has_Target_t = _has_Target_t<T>;
  

  template <typename T>
  concept _has_Iterator_t = requires(T t) {
    typename T::Iterator_t;
  };

  template <typename T>
  static constexpr bool has_Iterator_t = _has_Iterator_t<T>;

  template <typename T>
  concept _has_Inplace_t = requires(T t) {
    typename T::Inplace_t;
  };

  template <typename T>
  static constexpr bool has_Inplace_t = _has_Inplace_t<T>;

  template <typename T>
  concept _has_Vertex_Types = requires(T t) {
    T::Vertex_Types;
  };

  template <typename T>
  static constexpr bool has_Vertex_Types = _has_Vertex_Types<T>;

  template <typename T>
  concept _has_Edge_Types = requires(T t) {
    T::Edge_Types;
  };

  template <typename T>
  static constexpr bool has_Edge_Types = _has_Edge_Types<T>;

}  // namespace Sycl_Graph::Sycl

#endif