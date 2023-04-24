#ifndef SYCL_GRAPH_TYPE_HELPERS_HPP
#define SYCL_GRAPH_TYPE_HELPERS_HPP
#include <array>
#include <tuple>
#include <type_traits>
// #include <metal.hpp>
namespace Sycl_Graph {

  template <typename T, typename Tuple> struct has_type;

  template <typename T> struct has_type<T, std::tuple<>> : std::false_type {};

  template <typename T, typename U, typename... Ts> struct has_type<T, std::tuple<U, Ts...>>
      : has_type<T, std::tuple<Ts...>> {};

  template <typename T, typename... Ts> struct has_type<T, std::tuple<T, Ts...>> : std::true_type {
  };

  template <class T, class Tuple> struct Tuple_Index;

  template <class T, class... Types> struct Tuple_Index<T, std::tuple<T, Types...>> {
    static const std::size_t value = 0;
  };

  template <class T> struct Tuple_Index<T, std::tuple<>> {
    static const std::size_t value = 0;
  };
  template <class T, class U, class... Types> struct Tuple_Index<T, std::tuple<U, Types...>> {
    static const std::size_t value = 1 + Tuple_Index<T, std::tuple<Types...>>::value;
  };

  template <typename T, typename First, typename... Rest> constexpr auto index_of_type() {
    if constexpr (std::is_same_v<T, First>)
      return 0;
    else if constexpr (sizeof...(Rest) == 0)
      return -1;
    else
      return 1 + index_of_type<T, Rest...>();
  }

  template <typename... Ts, typename... Types>
  constexpr auto get_by_types(const std::tuple<Types...> &tuple) {
    return std::make_tuple(std::get<Ts>(tuple)...);
  }

  template <typename... Types> struct indices_of_types {
    template <typename... SubTypes> static constexpr auto get() {
      return std::make_tuple(index_of_type<SubTypes, Types...>()...);
    }
  };

  template <typename T, typename... Ts> struct is_in_pack {
    static constexpr bool value = std::disjunction_v<std::is_same<T, Ts>...>;
  };

  template <typename T, typename... Ts> struct Type_Map {
    std::array<T, sizeof...(Ts)> values;
    Type_Map() = default;
    Type_Map(const std::tuple<Ts...> &tuple) {}
    template <typename U> T get() { return values[index_of_type<U, Ts...>]; }
  };

  template <typename T, typename T_Array>
  concept Indexable = requires(const T_Array &array) {
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

  template <typename wanted_type, typename T> struct is_one_of {
    static constexpr bool value = std::is_same_v<wanted_type, T>;
  };

  template <typename wanted_type, typename... Ts> struct is_one_of<wanted_type, std::tuple<Ts...>> {
    static constexpr bool value = (std::is_same_v<wanted_type, Ts> || ...);
  };

  template <typename wanted_type, typename T> struct is_edge_of {
    static constexpr bool value = std::is_same_v<typename wanted_type::Edge_t, T>;
  };

  template <typename wanted_type, typename... Ts>
  struct is_edge_of<wanted_type, std::tuple<Ts...>> {
    static constexpr bool value
        = (std::is_same_v<typename wanted_type::Edge_t, typename Ts::Edge_t> || ...);
  };

  template <typename wanted_type, typename T> struct is_vertex_of {
    static constexpr bool value = std::is_same_v<typename wanted_type::Vertex_t, T>;
  };

  template <typename wanted_type, typename... Ts>
  struct is_vertex_of<wanted_type, std::tuple<Ts...>> {
    static constexpr bool value
        = (std::is_same_v<typename wanted_type::Vertex_t, typename Ts::Vertex_t> || ...);
  };

  template <typename T, typename... Ts> constexpr auto tuple_head(std::tuple<T, Ts...> t) {
    return std::get<0>(t);
  }

  template <std::size_t... Ns, typename... Ts>
  constexpr auto tuple_tail_impl(std::index_sequence<Ns...>, std::tuple<Ts...> t) {
    return std::make_tuple(std::get<Ns + 1u>(t)...);
  }

  template <typename... Ts> constexpr auto tuple_tail(std::tuple<Ts...> t) {
    return tuple_tail_impl(std::make_index_sequence<sizeof...(Ts) - 1u>(), t);
  }

  template <template <typename, typename...> typename Predicate, typename tuple_t, typename T>
  constexpr auto append_if_unique(const tuple_t &ut, const T &tuple_elem) {
    if constexpr (!Predicate<T, tuple_t>::value) {
      return std::tuple_cat(ut, std::make_tuple(tuple_elem));
    } else {
      return ut;
    }
  }

  template <template <typename, typename...> typename Predicate, typename tuple_t, typename... Ts>
  constexpr auto unique_tuple(const tuple_t &ut, const std::tuple<Ts...> &t);

  template <template <typename, typename...> typename Predicate, typename tuple_t, typename... Ts>
  constexpr auto unique_tuple(const tuple_t &ut, const std::tuple<Ts...> &t) {
    if constexpr (sizeof...(Ts) == 0) {
      return ut;
    } else {
      auto us_new = append_if_unique<Predicate>(ut, tuple_head(t));
      return unique_tuple<Predicate>(us_new, tuple_tail(t));
    }
  }

  template <template <typename, typename...> typename Predicate, typename tuple_t>
  constexpr auto unique_tuple(const tuple_t &ut) {
    return unique_tuple<Predicate>(std::make_tuple(tuple_head(ut)), tuple_tail(ut));
  }

  template <std::size_t i, typename tuple_element_t, typename wanted_element_t,
            template <typename, typename...> typename Predicate,
            bool wanted = Predicate<tuple_element_t, wanted_element_t>::value>
  struct extract_type {
    template <typename tuple_type> static auto do_extract_type(const tuple_type &t) {
      return std::tuple<>{};
    }
  };
  template <std::size_t i, typename tuple_element_t, typename wanted_element_t,
            template <typename, typename...> typename Predicate>
  struct extract_type<i, tuple_element_t, wanted_element_t, Predicate, true> {
    template <typename tuple_type> static auto do_extract_type(const tuple_type &t) {
      return std::tuple<tuple_element_t>{std::get<i>(t)};
    }
  };
  template <typename wanted_element_t, template <typename, typename...> typename Predicate,
            typename tuple_type, std::size_t... i>
  constexpr auto get_type_t(const tuple_type &t, std::index_sequence<i...>) {
    return std::tuple_cat(extract_type<i, typename std::tuple_element<i, tuple_type>::type,
                                       wanted_element_t, Predicate>::do_extract_type(t)...);
  }

  template <typename... wanted_element_t, template <typename, typename...> typename Predicate,
            typename... types>
  constexpr auto predicate_get_type(const std::tuple<types...> &t) {
    return get_type_t<std::tuple<wanted_element_t...>, Predicate>(
        t, std::make_index_sequence<sizeof...(types)>());
  }

  template <typename... wanted_element_t, typename... types>
  constexpr auto get_type(const std::tuple<types...> &t) {
    return get_type_t<std::tuple<wanted_element_t...>, is_one_of>(
        t, std::make_index_sequence<sizeof...(types)>());
  }

  template <typename... wanted_element_t, typename... types>
  constexpr auto get_type_with_edge(const std::tuple<types...> &t) {
    return get_type_t<std::tuple<wanted_element_t...>, is_vertex_of>(
        t, std::make_index_sequence<sizeof...(types)>());
  }

  template <typename... wanted_element_t, typename... types>
  constexpr auto get_type_with_vertex(const std::tuple<types...> &t) {
    return get_type_t<std::tuple<wanted_element_t...>, is_edge_of>(
        t, std::make_index_sequence<sizeof...(types)>());
  }

  template <template <typename, typename...> typename Predicate, typename... Ts>
  auto predicate_sort(const std::tuple<Ts...> &t) {
    auto unique_elems = unique_tuple<Predicate>(t);

    return std::apply(
        [&](auto... elem) {
          return std::make_tuple(get_type_t<std::tuple<decltype(elem)>, Predicate>(
              t, std::make_index_sequence<sizeof...(Ts)>())...);
        },
        unique_elems);
  }

  template <typename... Ts> constexpr auto type_sort(const std::tuple<Ts...> &t) {
    return predicate_sort<is_one_of>(t);
  }

  template <typename... Ts> constexpr auto edge_sort(const std::tuple<Ts...> &t) {
    return predicate_sort<is_edge_of>(t);
  }

  template <typename... Ts> constexpr auto vertex_sort(const std::tuple<Ts...> &t) {
    return predicate_sort<is_vertex_of>(t);
  }

  namespace detail {
    // Describe the type of a tuple with element I from each input tuple.
    // Needed to preserve the exact types from the input tuples.
    template <std::size_t I, typename... Tuples> using zip_tuple_at_index_t
        = std::tuple<std::tuple_element_t<I, std::decay_t<Tuples>>...>;

    // Collect all elements at index I from all input tuples as a new tuple.
    template <std::size_t I, typename... Tuples>
    zip_tuple_at_index_t<I, Tuples...> zip_tuple_at_index(Tuples &&...tuples) {
      return {std::get<I>(std::forward<Tuples>(tuples))...};
    }

    // Create a tuple with the result of zip_tuple_at_index for each index.
    // The explicit return type prevents flattening into a single tuple
    // when sizeof...(Tuples) == 1 or sizeof...(I) == 1 .
    template <typename... Tuples, std::size_t... I>
    std::tuple<zip_tuple_at_index_t<I, Tuples...>...> tuple_zip_impl(Tuples &&...tuples,
                                                                     std::index_sequence<I...>) {
      return {zip_tuple_at_index<I>(std::forward<Tuples>(tuples)...)...};
    }

  }  // namespace detail

  // Zip a number of tuples together into a tuple of tuples.
  // Take the first tuple separately so we can easily get its size.
  template <typename Head, typename... Tail> auto tuple_zip(Head &&head, Tail &&...tail) {
    constexpr std::size_t size = std::tuple_size_v<std::decay_t<Head>>;

    static_assert(((std::tuple_size_v<std::decay_t<Tail>> == size) && ...),
                  "Tuple size mismatch, can not zip.");

    return detail::tuple_zip_impl<Head, Tail...>(
        std::forward<Head>(head), std::forward<Tail>(tail)..., std::make_index_sequence<size>());
  }

}  // namespace Sycl_Graph
#endif