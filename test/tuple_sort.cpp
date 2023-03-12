#include <iostream>
#include <tuple>
#include <type_traits>
#include <utility>
#include <Sycl_Graph/Buffer/Sycl/type_helpers.hpp>
// Answer one simple question: here's a type, and a tuple. Tell me
// if the type is one of the tuples types. If so, I want it.

// template <typename wanted_type, typename T> struct is_wanted_type;

// template <typename wanted_type, typename... Types>
// struct is_wanted_type<wanted_type, std::tuple<Types...>> {
//   static constexpr bool wanted = (std::is_same_v<wanted_type, Types> || ...);
// };

// // Ok, the ith index in the tuple, here's its std::tuple_element type.
// // And wanted_element_t is a tuple of all types we want to extract.
// //
// // Based on which way the wind blows we'll produce either a std::tuple<>
// // or a std::tuple<tuple_element_t>.

// template <size_t i, typename tuple_element_t, typename wanted_element_t,
//           bool wanted = is_wanted_type<tuple_element_t, wanted_element_t>::wanted>
// struct extract_type {
//   template <typename tuple_type> static auto do_extract_type(const tuple_type &t) {
//     return std::tuple<>{};
//   }
// };
// template <size_t i, typename tuple_element_t, typename wanted_element_t>
// struct extract_type<i, tuple_element_t, wanted_element_t, true> {
//   template <typename tuple_type> static auto do_extract_type(const tuple_type &t) {
//     return std::tuple<tuple_element_t>{std::get<i>(t)};
//   }
// };

// template <typename wanted_element_t, typename tuple_type, size_t... i>
// auto get_type_t(const tuple_type &t, std::index_sequence<i...>) {
//   return std::tuple_cat(extract_type<i, typename std::tuple_element<i, tuple_type>::type,
//                                      wanted_element_t>::do_extract_type(t)...);
// }

// template <typename... wanted_element_t, typename... types>
// auto get_type(const std::tuple<types...> &t) {
//   return get_type_t<std::tuple<wanted_element_t...>>(t,
//                                                      std::make_index_sequence<sizeof...(types)>());
// }

// gets one element of each type

// template <typename wanted_type, typename T> struct is_one_of
// {
//   static constexpr bool value = std::is_same_v<wanted_type, T>;
// };


// template <typename wanted_type, typename... Ts> struct is_one_of<wanted_type, std::tuple<Ts...>> {
//   static constexpr bool value = (std::is_same_v<wanted_type, Ts> || ...);
// };

// template <typename wanted_type, typename T> struct is_edge_of
// {
//   static constexpr bool value = std::is_same_v<typename wanted_type::Edge_t, T>;
// };

// template <typename wanted_type, typename... Ts> struct is_edge_of<wanted_type, std::tuple<Ts...>> {
//   static constexpr bool value = (std::is_same_v<typename wanted_type::Edge_t, typename Ts::Edge_t> || ...);
// };

// template <typename wanted_type, typename T> struct is_vertex_of
// {
//   static constexpr bool value = std::is_same_v<typename wanted_type::Vertex_t, T>;
// };

// template <typename wanted_type, typename... Ts>
// struct is_vertex_of<wanted_type, std::tuple<Ts...>> {
//   static constexpr bool value = (std::is_same_v<typename wanted_type::Vertex_t, typename Ts::Vertex_t> || ...);
// };

// template < typename T , typename... Ts >
// constexpr auto tuple_head( std::tuple<T,Ts...> t )
// {
//    return  std::get<0>(t);
// }

// template < std::size_t... Ns , typename... Ts >
// constexpr auto tuple_tail_impl( std::index_sequence<Ns...> , std::tuple<Ts...> t )
// {
//    return  std::make_tuple( std::get<Ns+1u>(t)... );
// }

// template < typename... Ts >
// constexpr auto tuple_tail( std::tuple<Ts...> t )
// {
//    return  tuple_tail_impl( std::make_index_sequence<sizeof...(Ts) - 1u>() , t );
// }

// template <template <typename, typename...> typename Predicate, typename tuple_t, typename T>
// constexpr auto append_if_unique(const tuple_t &ut, const T &tuple_elem) {
//   if constexpr (!Predicate<T, tuple_t>::value) {
//     return std::tuple_cat(ut, std::make_tuple(tuple_elem));
//   } else {
//     return ut;
//   }
// }

// template <template <typename, typename...> typename Predicate, typename tuple_t, typename... Ts>
// constexpr auto unique_tuple(const tuple_t &ut, const std::tuple<Ts...> &t);

// template <template <typename, typename...> typename Predicate, typename tuple_t, typename... Ts>
// constexpr auto unique_tuple(const tuple_t &ut, const std::tuple<Ts...> &t) {
//   if constexpr (sizeof...(Ts) == 0) {
//     return ut;
//   } else {
//     auto us_new = append_if_unique<Predicate>(ut, tuple_head(t));
//     return unique_tuple<Predicate>(us_new, tuple_tail(t));
//   }
// }

// template <template <typename, typename...> typename Predicate, typename tuple_t>
// constexpr auto unique_tuple(const tuple_t &ut) {
//   return unique_tuple<Predicate>(std::make_tuple(tuple_head(ut)), tuple_tail(ut));
// }


// template <size_t i, typename tuple_element_t, typename wanted_element_t,
//           template <typename, typename...> typename Predicate,
//           bool wanted = Predicate<tuple_element_t, wanted_element_t>::value>
// struct extract_type {
//   template <typename tuple_type> static auto do_extract_type(const tuple_type &t) {
//     return std::tuple<>{};
//   }
// };
// template <size_t i, typename tuple_element_t, typename wanted_element_t,
//           template <typename, typename...> typename Predicate>
// struct extract_type<i, tuple_element_t, wanted_element_t, Predicate, true> {
//   template <typename tuple_type> static auto do_extract_type(const tuple_type &t) {
//     return std::tuple<tuple_element_t>{std::get<i>(t)};
//   }
// };
// template <typename wanted_element_t, template <typename, typename...> typename Predicate,
//           typename tuple_type, size_t... i>
// constexpr auto get_type_t(const tuple_type &t, std::index_sequence<i...>) {
//   return std::tuple_cat(extract_type<i, typename std::tuple_element<i, tuple_type>::type,
//                                      wanted_element_t, Predicate>::do_extract_type(t)...);
// }

// template <typename... wanted_element_t, template <typename, typename...> typename Predicate,
//           typename... types>
// constexpr auto predicate_get_type(const std::tuple<types...> &t) {
//   return get_type_t<std::tuple<wanted_element_t...>, Predicate>(
//       t, std::make_index_sequence<sizeof...(types)>());
// }


// template <typename... wanted_element_t, typename... types>
// constexpr auto get_type(const std::tuple<types...> &t) {
//   return get_type_t<std::tuple<wanted_element_t...>, is_one_of>(
//       t, std::make_index_sequence<sizeof...(types)>());
// }

// template <typename... wanted_element_t, typename... types>
// constexpr auto get_type_with_edge(const std::tuple<types...> &t) {
//   return get_type_t<std::tuple<wanted_element_t...>, is_vertex_of>(
//       t, std::make_index_sequence<sizeof...(types)>());
// }

// template <typename... wanted_element_t, typename... types>
// constexpr auto get_type_with_vertex(const std::tuple<types...> &t) {
//   return get_type_t<std::tuple<wanted_element_t...>, is_edge_of>(
//       t, std::make_index_sequence<sizeof...(types)>());
// }


// template <template <typename, typename...> typename Predicate, typename... Ts> 
// auto predicate_sort(const std::tuple<Ts...> &t) {
//   auto unique_elems = unique_tuple<Predicate>(t);

//   return std::apply(
//       [&](auto... elem) {
//         return std::make_tuple(
//             get_type_t<std::tuple<decltype(elem)>, Predicate>(
//       t, std::make_index_sequence<sizeof...(Ts)>()) ...);
//       },
//       unique_elems);
// }

// template <typename... Ts> constexpr auto type_sort(const std::tuple<Ts...> &t) {
//   return predicate_sort<is_one_of>(t);
// }


// template <typename... Ts> constexpr auto edge_sort(const std::tuple<Ts...> &t) {
//   return predicate_sort<is_edge_of>(t);
// }

// template <typename... Ts> constexpr auto vertex_sort(const std::tuple<Ts...> &t) {
//   return predicate_sort<is_vertex_of>(t);
// }


struct Foo {
  typedef double Vertex_t;
  typedef double Edge_t;
};
struct Bar {
  typedef int Vertex_t;
  typedef int Edge_t;
};
using namespace Sycl_Graph;
int main() {
  std::tuple<Foo, Bar, Foo, Bar, Bar> t;
  // auto x = Vertex::get_type<Foo::Vertex_t>(t);
  // auto x_edge = Edge::get_type<Bar::Edge_t>(t);
  auto x0 = get_type_with_vertex<Foo>(t);
  auto xu = unique_tuple<is_one_of>(t);
  auto xe = edge_sort(t);  

  std::tuple<sycl::buffer<int, 1>, sycl::buffer<double, 1>, sycl::buffer<double>> buffers = {sycl::buffer<int, 1>(sycl::range<1>(10)), sycl::buffer<double, 1>(sycl::range<1>(10)), sycl::buffer<double>(sycl::range<1>(10))};
  auto xb = Sycl_Graph::Sycl::buffer_sort(buffers);

  //print xb size
  std::cout << std::tuple_size<decltype(xb)>::value << std::endl;

  auto xb0 = std::get<0>(xb);
  auto xb1 = std::get<1>(xb);

  //print xb 0 size
  std::cout << std::tuple_size<decltype(xb0)>::value << std::endl;

  //print xb 1 size
  std::cout << std::tuple_size<decltype(xb1)>::value << std::endl;

  std::tuple <double, double, float> a;
  std::tuple <int, int, std::size_t> b;
  std::tuple <char, char, char> c;

  auto d = tuple_zip(a, b, c);


  return 0;
}