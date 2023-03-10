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
      static constexpr bool value = std::is_same_v<typename T0::Edge_Buffer_t, typename T1::Edge_Buffer_t>;
    };
  } // namespace _detail

template <typename T, typename... Ts>
struct unique_tuple : std::type_identity<T> {};

template <typename... Ts, typename U, typename... Us>
struct unique_tuple<std::tuple<Ts...>, U, Us...>
    : std::conditional_t<(std::is_same_v<U, Ts> || ...)
                       , unique_tuple<std::tuple<Ts...>, Us...>
                       , unique_tuple<std::tuple<Ts..., U>, Us...>> {};


template <template <typename, typename> typename  Predicate, typename... Ts>
constexpr auto separate_by_type(std::tuple<Ts...> types) {
    using Unique_Types = typename unique_tuple<Ts ...>::type;
      return 
        std::apply(
            [](auto... ts) {
              return std::tuple_cat(
                  std::conditional_t<Predicate<Unique_Types, decltype(ts)>::value, std::tuple<decltype(ts)>, std::tuple<>>::type ...);
            },
            types);
}

template <typename ... Ts>
constexpr auto unique_types(std::tuple<Ts...> types) {
  return separate_by_type<std::is_same>(types);
}

template <typename ... Ts>
constexpr auto unique_type_by_subtype(std::tuple<Ts...> types) {
  return separate_by_type<std::is_same>(types);
}

template <template <typename, typename> typename  Predicate, template <typename, typename> typename UniquePredicate, typename... Ts>
constexpr auto separate_by_subtype(std::tuple<Ts...> types) {
  constexpr auto unique_member_types = separate_by_type<UniquePredicate>(types);

  return std::apply(
      [&](auto... unique_type) {return 
        std::apply(
            [](auto... ts) {
              return std::tuple_cat(
                  std::conditional_t<Predicate<decltype(ts), decltype(unique_type)>::value, std::tuple<decltype(ts)>, std::tuple<>>::type ...);
            },
            types);
      },
      unique_member_types);
}
template <typename... Ts> constexpr auto separate_by_edge_type(const std::tuple<Ts...>& types) {
  return separate_by_subtype<_detail::_Has_Same_Edge, _detail::_Has_Same_Edge>(types);
}

template <typename... Ts> constexpr auto separate_by_vertex_type(const std::tuple<Ts...>& types) {
  return separate_by_subtype<_detail::_Has_Same_Vertex, _detail::_Has_Same_Vertex>(types);
}

template <typename... Ts> constexpr auto separate_by_type(const std::tuple<Ts...>& types) {
  return separate_by_type<std::is_same>(types);
}
}  // namespace Sycl_Graph::Sycl

#endif