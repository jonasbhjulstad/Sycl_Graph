#ifndef SYCL_GRAPH_ALGORITHMS_PROPERTIES_SYCL_PROPERTY_EXTRACTOR_HPP
#define SYCL_GRAPH_ALGORITHMS_PROPERTIES_SYCL_PROPERTY_EXTRACTOR_HPP

#include <CL/sycl.hpp>
#include <Sycl_Graph/Graph/Sycl/Invariant/Graph.hpp>
#include <Sycl_Graph/Algorithms/Properties/Invariant/Property_Extractor.hpp>
#include <Sycl_Graph/Buffer/Sycl/type_helpers.hpp>
#include <tuple>

namespace Sycl_Graph::Sycl {


  template <Sycl_Graph::Sycl::Invariant::Graph_type Graph_t, Sycl_Graph::Invariant::Property_Extractor_type... Es>
  sycl::event single_type_extractor_apply(
      Graph_t& graph, const std::tuple<Es...>& extractors,
      std::tuple<sycl::buffer<typename Es::Property_t>...>& apply_bufs, sycl::queue& q) {
    using Edge_t = typename std::tuple_element<0, std::tuple<Es...>>::type::Edge_t;
    using From_t = typename Edge_t::From_t;
    using To_t = typename Edge_t::To_t;
    return q.submit([&](sycl::handler& h) {
      std::apply(
          [&](auto&... apply_buf) {
            return std::make_tuple(apply_buf.template get_access<sycl::access::mode::read>(h)...);
          },
          apply_bufs);
      auto to_acc = graph.template get_access<sycl::access::mode::read, To_t>(h);
      auto from_acc = graph.template get_access<sycl::access::mode::read, From_t>(h);
      auto edge_acc = graph.template get_access<sycl::access::mode::read, Edge_t>(h);
      auto apply_acc = std::apply(
          [&](auto&... apply_buf) {
            return std::make_tuple(apply_buf.template get_access<sycl::access::mode::write>(h)...);
          },
          apply_bufs);

      // h.parallel_for(edge_acc.size(), [=](sycl::id<1> i) {
      //   const Edge_t edge = edge_acc[i];
      //   const From_t& from_vertex = from_acc[edge.from];
      //   const To_t& to_vertex = to_acc[edge.to];
      //   std::apply(
      //       [&](auto... ex) {
      //         std::apply([&](auto&... ap) { ((apply_acc[i] = ex.apply(edge, from_vertex, to_vertex)), ...); },
      //                    apply_acc);
      //       },
      //       extractors);
      // });
    });
  }

  template <Sycl_Graph::Sycl::Invariant::Graph_type Graph_t, Sycl_Graph::Invariant::Property_Extractor_type... Es>
  sycl::event single_type_extractor_accumulate(
      Graph_t& graph, const std::tuple<Es...>& extractors,
      std::tuple<sycl::buffer<typename Es::Property_t>...>& apply_bufs,
      std::tuple<sycl::buffer<typename Es::Accumulation_Property_t>...>& accumulate_bufs,
      sycl::queue& q, sycl::event& apply_event) {
    using Edge_t = typename std::tuple_element<0, std::tuple<Es...>>::type::Edge_t;
    return q.submit([&](sycl::handler& h) {
      h.depends_on(apply_event);
      auto apply_acc = std::apply(
          [&](auto&... apply_buf) {
            return std::make_tuple(apply_buf.template get_access<sycl::access::mode::read>(h)...);
          },
          apply_bufs);
      auto accumulate_acc = std::apply(
          [&](auto&... accumulate_buf) {
            return std::make_tuple(
                accumulate_buf.template get_access<sycl::access::mode::write>(h)...);
          },
          accumulate_bufs);
      auto edge_acc = graph.edge_buf.template get_access<sycl::access::mode::read, Edge_t>(h);
      auto to_acc
          = graph.vertex_buf.template get_access<sycl::access::mode::read, typename Edge_t::To_t>(
              h);
      auto from_acc
          = graph.vertex_buf.template get_access<sycl::access::mode::read, typename Edge_t::From_t>(
              h);

      h.single_task([&]() {
        std::apply(
            [&](auto&... ap) {
              std::apply(
                  [&](auto... ex) {
                    std::apply(
                        [&](auto&... ac) {
                          ((ac[0] = ex.accumulate(edge_acc, from_acc, to_acc, apply_acc)), ...);
                        },
                        accumulate_acc);
                  },
                  extractors);
            },
            apply_acc);
      });
    });
  }

  template <typename wanted_type, typename T0> 
  struct is_edge_of_extractor {
    template <typename T> using extractor_t = typename std::tuple_element_t<0, T>;
    static constexpr bool value = std::is_same_v<typename extractor_t<T0>::Edge_t,
                                                 typename extractor_t<wanted_type>::Edge_t>;
  };


  template <typename wanted_type, typename... Extractor_Buffer_Zip>
  struct is_edge_of_extractor<wanted_type, ::std::tuple<Extractor_Buffer_Zip...>> {
    template <typename T> using extractor_t = std::tuple_element_t<0, T>;
    static constexpr bool value
        = (std::is_same_v<typename extractor_t<Extractor_Buffer_Zip>::Edge_t,
                          typename extractor_t<wanted_type>::Edge_t>
           || ...);
  };

  template <typename... Ts> constexpr auto extractor_edge_sort(const std::tuple<Ts...>& t) {
    return predicate_sort<is_edge_of_extractor>(t);
  }

  template <Sycl_Graph::Sycl::Invariant::Graph_type Graph_t, Sycl_Graph::Invariant::Property_Extractor_type... Es>
  auto extractor_apply(Graph_t& graph, const std::tuple<Es...>& extractors,
                       std::tuple<sycl::buffer<typename Es::Property_t>...>& apply_buf,
                       sycl::queue& q) {
    // sort extractor/buf-zip to obtain tuple: {[(extractor, buf), (extractor, buf), ...], [(ex...),
    // (buf...)]}
    auto edge_sorted_pack = extractor_edge_sort(tuple_zip(extractors, apply_buf));

    // zip to obtain tuple: {[(extractor, extractor, ...), (buf, buf, ...)], [(ex...), (buf...)]}
    auto extractor_buf_sorted_pack = std::apply([&](const auto&... es_pack) {
              return std::make_tuple(std::apply(
                  [&](const auto&... bs_pack) {
                    return std::make_tuple(std::make_tuple(std::get<0>(bs_pack) ...), std::make_tuple(std::get<1>(bs_pack) ...));
                  },
                  es_pack) ...
              );
    }, edge_sorted_pack);

    auto apply_events = std::apply(
        [&](auto&... es_pack) {
          return std::make_tuple(
              (single_type_extractor_apply(graph, std::get<0>(es_pack), std::get<1>(es_pack),
              q))...);
        },
        extractor_buf_sorted_pack);
    // auto apply_events = 0;
    return apply_events;
  }

  template <Sycl_Graph::Sycl::Invariant::Graph_type Graph_t, Sycl_Graph::Invariant::Property_Extractor_type... Es>
  auto extractor_accumulate(
      Graph_t& graph, const std::tuple<Es...>& extractors,
      std::tuple<sycl::buffer<typename Es::Property_t>...>& apply_buf,
      std::tuple<sycl::buffer<typename Es::Accumulation_Property_t>...>& accumulate_buf,
      sycl::queue& q, auto& apply_events) {
    auto edge_sorted_pack = extractor_edge_sort(tuple_zip(extractors, apply_buf, accumulate_buf));
    auto accumulate_events = std::apply(
        [&](auto... es_pack) {
          return std::make_tuple(
              (single_type_extractor_accumulate(graph, std::get<0>(es_pack), std::get<1>(es_pack),
                                                std::get<2>(es_pack), q),
               ...));
        },
        edge_sorted_pack);
    return accumulate_events;
  }
  template <Sycl_Graph::Sycl::Invariant::Graph_type Graph_t, Sycl_Graph::Invariant::Property_Extractor_type... Es>
  std::tuple<sycl::buffer<typename Es::Property_t>...> construct_apply_buffers(
      Graph_t& graph, const std::tuple<Es...>& extractors) {
    auto edge_sizes = std::apply(
        [&](auto... extractor) {
          return std::make_tuple(graph.template current_size<typename Es::Edge_t>()...);
        },
        extractors);

    std::tuple<sycl::buffer<typename Es::Property_t>...> bufs = std::apply(
        [&](auto&... edge_size) {
          return std::make_tuple(sycl::buffer<typename Es::Property_t>(edge_size)...);
        },
        edge_sizes);

    return bufs;
  }

  template <Sycl_Graph::Sycl::Invariant::Graph_type Graph_t, Sycl_Graph::Invariant::Property_Extractor_type... Es>
  std::tuple<sycl::buffer<typename Es::Accumulation_Property_t>...> construct_accumulation_buffers(
      Graph_t& graph, const std::tuple<Es...>& extractors) {
    auto edge_sizes = std::apply(
        [&](auto... extractor) {
          return std::make_tuple(graph.edge_buf.template current_size<typename Es::Edge_t>()...);
        },
        extractors);
    std::tuple<sycl::buffer<typename Es::Accumulation_Property_t>...> bufs = std::apply(
        [&](auto&... edge_size) {
          return std::apply(
              [&](auto... extractors) {
                return std::make_tuple(
                    sycl::buffer<typename Es::Accumulation_Property_t>(edge_size)...);
              },
              extractors);
        },
        edge_sizes);
    return bufs;
  }

  template <Sycl_Graph::Sycl::Invariant::Graph_type Graph_t, Sycl_Graph::Invariant::Property_Extractor_type... Es>
  std::tuple<std::vector<typename Es::Accumulation_Property_t>...> extract_properties(
      Graph_t& graph, const std::tuple<Es...>& extractors, sycl::queue& q) {
    std::tuple<sycl::buffer<typename Es::Property_t>...> apply_buffers
        = construct_apply_buffers(graph, extractors);
    auto accumulate_buffers = construct_accumulation_buffers(graph, extractors);

    auto apply_events = extractor_apply(graph, extractors, apply_buffers, q);
    // auto accumulate_events = extractor_accumulate(graph, extractors, apply_buffers,
    //                                               accumulate_buffers, q, apply_events);
    // auto accumulate_acc = std::apply(
    //     [&](auto... accumulate_bufs) {
    //       return std::make_tuple(
    //           (accumulate_bufs.template get_access<sycl::access::mode::read>(q), ...));
    //     },
    //     accumulate_buffers);

    // TODO: for-loop to extract all properties
    q.wait_and_throw();

    return {};
  }


}  // namespace Sycl_Graph::Sycl
#endif