#ifndef SYCL_GRAPH_BUFFER_SYCL_TYPE_HELPERS_HPP
#define SYCL_GRAPH_BUFFER_SYCL_TYPE_HELPERS_HPP
#include <CL/sycl.hpp>
#include <Sycl_Graph/type_helpers.hpp>

namespace Sycl_Graph::Sycl {

  // get data type of sycl buffer
  template <typename T> struct Buffer_Data_Type;

  template <typename T, int D> struct Buffer_Data_Type<cl::sycl::buffer<T, D>> {
    using type = T;
  };

  template <typename wanted_type, typename T> struct is_buffer_of {
    static constexpr bool value = std::is_same_v<typename Buffer_Data_Type<T>::type, typename Buffer_Data_Type<wanted_type>::type>;
  };


  template <typename wanted_type, typename... Ts>
  struct is_buffer_of<wanted_type, std::tuple<Ts...>> {
    static constexpr bool value = (std::is_same_v<typename Buffer_Data_Type<Ts>::type, typename Buffer_Data_Type<wanted_type>::type> || ...);
  };

  template <typename... Ts> constexpr auto buffer_sort(const std::tuple<Ts...> &t) {
    return predicate_sort<is_buffer_of>(t);
  }

}  // namespace Sycl_Graph::Sycl

#endif