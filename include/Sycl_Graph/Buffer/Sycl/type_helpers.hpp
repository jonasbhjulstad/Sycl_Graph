#ifndef SYCL_GRAPH_BUFFER_SYCL_TYPE_HELPERS_HPP
#define SYCL_GRAPH_BUFFER_SYCL_TYPE_HELPERS_HPP
#include <CL/sycl.hpp>
#include <Sycl_Graph/type_helpers.hpp>

namespace Sycl_Graph::Sycl {

  namespace _detail {
    template <typename T0, typename T1> struct _Has_Same_Edge {
      static constexpr bool value = std::is_same_v<typename T0::Edge_t, typename T1::Edge_t>;
    };

    template <typename T0, typename T1> struct _Has_Same_Vertex {
      static constexpr bool value = std::is_same_v<typename T0::Vertex_t, typename T1::Vertex_t>;
    };

    template <typename T0, typename T1> struct _Has_Same_Edge_Buf {
      static constexpr bool value = std::is_same_v<typename T0::Edge_Buffer_t, typename T1::>;
    };
  } // namespace _detail

template <template <typename, typename> Predicate, typename... Ts>
constexpr auto separate_by_type(std::tuple<Ts...> types) {
  constexpr auto unique_member_types = unique_types(types);

  return std::apply(
      [&](auto... unique_types) {
        std::apply(
            [](auto... ts) {
              return std::tuple_cat(
                  std::conditional_t<
                      (std::is_same_v<Predicate<decltype(unique_types), decltype(types)>::value>),
                      std::tuple<decltype(ts)>, std::tuple<>>{}...);
            },
            types);
      },
      unique_member_types);
}

template <typename... Ts> constexpr auto separate_by_edge_type(std::tuple<Ts...>& types) {
  return separate_by_type<_detail::_Has_Same_Edge>(types);
}

template <typename... Ts> constexpr auto separate_by_vertex_type(std::tuple<Ts...>& types) {
  return separate_by_type<_detail::_Has_Same_Vertex>(types);
}

template <typename... Ts> constexpr auto separate_by_type(std::tuple<Ts...>& types) {
  return separate_by_type<std::is_same>(types);
}
}  // namespace Sycl_Graph::Sycl

#endif