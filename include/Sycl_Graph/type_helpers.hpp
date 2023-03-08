#ifndef SYCL_GRAPH_TYPE_HELPERS_HPP
#define SYCL_GRAPH_TYPE_HELPERS_HPP
#include <array>
#include <tuple>
#include <type_traits>
// #include <metal.hpp>
namespace Sycl_Graph {
  template <class T, class Tuple> struct Tuple_Index;

  template <class T, class... Types> struct Tuple_Index<T, std::tuple<T, Types...>> {
    static const std::size_t value = 0;
  };

  template <class T, class U, class... Types> struct Tuple_Index<T, std::tuple<U, Types...>> {
    static const std::size_t value = 1 + Tuple_Index<T, std::tuple<Types...>>::value;
  };



template <typename T, typename First, typename ... Rest>
constexpr auto index_of_type() {
  if constexpr (sizeof...(Rest) == 0)
    return 0;
  else
    return 1 + index_of_type<T, Rest...>();
}

  template <typename... Ts, typename... Types>
  constexpr auto get_by_types(const std::tuple<Types...>& tuple) {
    return std::make_tuple(std::get<index_of_type<Ts, Types>>(tuple)...);
  }

  template <typename T, typename... Ts> struct is_in_pack {
    static constexpr bool value = std::disjunction_v<std::is_same<T, Ts>...>;
  };

  template <typename T, typename... Ts> struct Type_Map {
    std::array<T, sizeof...(Ts)> values;
    Type_Map() = default;
    Type_Map(const std::tuple<Ts...>& tuple) {}
    template <typename U> T get() { return values[index_of_type<U, Ts...>]; }
  };

  template <typename T, typename T_Array>
  concept Indexable = requires(const T_Array& array) {
                        { array[0] } -> std::same_as<T>;
                      };

  template <typename T> struct TypeBase {};

  template <typename... Ts> struct TypeSet : TypeBase<Ts>... {
    static constexpr std::tuple<Ts...> types{};
    template <typename T> constexpr auto operator+(TypeBase<T>) {
      if constexpr (std::is_base_of_v<TypeBase<T>, TypeSet>)
        return TypeSet{};
      else
        return TypeSet<Ts..., T>{};
    }

    constexpr auto size() const -> std::size_t { return sizeof...(Ts); }
  };

  template <typename... Ts> constexpr auto unique_types(std::tuple<Ts...> types) {
    return TypeSet<Ts...>::types;
  }

  template <typename... Ts> constexpr auto unique_types() { return TypeSet<Ts...>::types; }

  template <typename... Ts> constexpr auto filter_types(std::tuple<Ts...>& t) {
    return std::apply(
        [](auto... ts) {
          return std::tuple_cat(std::conditional_t<(decltype(ts)::value > 3),
                                                   std::tuple<decltype(ts)>, std::tuple<>>{}...);
        },
        t);
  }


  // template <template <typename> Predicate_t, typename... Ts>
  // constexpr auto separate_types(std::tuple<Ts...> types) {
  //   return std::apply(
  //       [](auto... ts) {
  //         return std::tuple_cat(std::conditional_t<(Predicate_t<decltype(ts)>::value),
  //                                                  std::tuple<decltype(ts)>, std::tuple<>>{}...);
  //       },
  //       types);
  //   // create one tuple for each unique type
  // }


}  // namespace Sycl_Graph
#endif