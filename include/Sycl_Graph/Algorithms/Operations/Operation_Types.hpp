#ifndef SYCL_GRAPH_ALGORITHMS_PROPERTIES_Operation_TYPES_HPP
#define SYCL_GRAPH_ALGORITHMS_PROPERTIES_Operation_TYPES_HPP
#include <CL/sycl.hpp>
#include <Sycl_Graph/Graph/Base/Graph_Types.hpp>
#include <concepts>
#include <type_traits>
namespace Sycl_Graph::Sycl {

  template <typename T>
  concept has_Graph_Iterator = true;

  template <typename T>
  concept Operation_type = true;

  template <typename T>
  concept Transform_Operation_type = Operation_type<T> && requires(T op) {
                                                            typename T::Source_t;
                                                            typename T::Target_t;
                                                            typename T::target_access_mode;
                                                          };
  template <typename T> constexpr bool is_Transform_Operation_type = Transform_Operation_type<T>;

  template <typename T>
  concept Injection_Operation_type = Operation_type<T> && requires(T op) {
                                                            is_Graph_element<typename T::Target_t>;
                                                            typename T::Target_t;
                                                          };
  template <typename T> constexpr bool is_Injection_Operation_type = Injection_Operation_type<T>;

  template <typename T>
  concept Extraction_Operation_type = Operation_type<T> && requires(T op) {
                                                             is_Graph_element<typename T::Source_t>;
                                                             typename T::Source_t;
                                                           };
  template <typename T> constexpr bool is_Extraction_Operation_type = Extraction_Operation_type<T>;

  template <Operation_type Op> using Source_Buffer_t
      = std::optional<sycl::buffer<typename Op::Source_t>>;

  template <Operation_type Op> using Target_Buffer_t
      = std::optional<sycl::buffer<typename Op::Target_t>>;

}  // namespace Sycl_Graph::Sycl

#endif