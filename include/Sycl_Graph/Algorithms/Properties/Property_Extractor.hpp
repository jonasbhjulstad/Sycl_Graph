#ifndef SYCL_GRAPH_ALGORITHMS_PROPERTIES_SYCL_PROPERTY_EXTRACTOR_HPP
#define SYCL_GRAPH_ALGORITHMS_PROPERTIES_SYCL_PROPERTY_EXTRACTOR_HPP

#include <CL/sycl.hpp>
#include <Sycl_Graph/Buffer/Sycl/type_helpers.hpp>
#include <Sycl_Graph/Graph/Sycl/Invariant_Graph.hpp>
#include <tuple>

namespace Sycl_Graph {
  template <typename T>
  concept Property_Extractor_type = true;

  template <Sycl_Graph::Sycl::Graph_type Graph_t,
            Sycl_Graph::Property_Extractor_type Es>
  sycl::event single_extractor_apply(Graph_t& graph, const Es& extractor,
                                     sycl::buffer<typename Es::Apply_t>& apply_buf,
                                     sycl::queue& q) {
    using Edge_t = typename Es::Edge_t;
    using From_t = typename Edge_t::From_t;
    using To_t = typename Edge_t::To_t;
    auto& bufs = graph.edge_buf.buffers;
    using Buf_t = decltype(std::get<0>(bufs));
    return q.submit([&](sycl::handler& h) {
      auto apply_acc = apply_buf.template get_access<sycl::access::mode::write>(h);
      auto edge_acc = graph.template get_edge_access<sycl::access::mode::read, Edge_t>(h);
      auto from_acc = graph.template get_vertex_access<sycl::access::mode::read, From_t>(h);
      auto to_acc = graph.template get_vertex_access<sycl::access::mode::read, To_t>(h);
      h.parallel_for(edge_acc.size(), [=](sycl::id<1> i) {
        const Edge_t& edge = edge_acc[i];
        apply_acc[i] = extractor.apply(edge, from_acc[edge.from], to_acc[edge.to]);
      });
    });
  }

  template <Sycl_Graph::Sycl::Graph_type Graph_t,
            Sycl_Graph::Property_Extractor_type Es>
  sycl::event single_extractor_accumulate(Graph_t& graph, const Es& extractor,
                                          sycl::buffer<typename Es::Apply_t>& apply_buf,
                                          sycl::buffer<typename Es::Accumulate_t>& accumulate_buf,
                                          sycl::queue& q, sycl::event& apply_event) {
    using Edge_t = typename Es::Edge_t;
    return q.submit([&](sycl::handler& h) {
      h.depends_on(apply_event);
      auto apply_acc = apply_buf.template get_access<sycl::access::mode::read>(h);
      auto accumulate_acc = accumulate_buf.template get_access<sycl::access::mode::read_write>(h);
      h.parallel_for(graph.N_vertices(),
                     [=](sycl::id<1> id) { extractor.accumulate(apply_acc, accumulate_acc, id); });
    });
  }

  template <typename wanted_type, typename T0> struct is_edge_of_extractor {
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

  template <Sycl_Graph::Sycl::Graph_type Graph_t,
            Sycl_Graph::Property_Extractor_type... Es>
  auto extractor_apply(Graph_t& graph, const std::tuple<Es...>& extractors,
                       std::tuple<sycl::buffer<typename Es::Apply_t>...>& apply_bufs,
                       sycl::queue& q) {
    return std::apply(
        [&](auto&... apply_b) {
          return std::apply(
              [&](const auto&... ex) {
                return std::make_tuple(single_extractor_apply(graph, ex, apply_b, q)...);
              },
              extractors);
        },
        apply_bufs);
  }

  template <Sycl_Graph::Sycl::Graph_type Graph_t,
            Sycl_Graph::Property_Extractor_type... Es>
  auto extractor_accumulate(Graph_t& graph, const std::tuple<Es...>& extractors,
                            std::tuple<sycl::buffer<typename Es::Apply_t>...>& apply_bufs,
                            std::tuple<sycl::buffer<typename Es::Accumulate_t>...>& accumulate_bufs,
                            sycl::queue& q, auto& apply_events) {
    return std::apply(
        [&](auto&... apply_event) {
          return std::apply(
              [&](auto&... accumulate_buf) {
                return std::apply(
                    [&](auto&... apply_buf) {
                      return std::apply(
                          [&](const auto&... ex) {
                            return std::make_tuple(single_extractor_accumulate(
                                graph, ex, apply_buf, accumulate_buf, q, apply_event)...);
                          },
                          extractors);
                    },
                    apply_bufs);
              },
              accumulate_bufs);
        },
        apply_events);
  }

  template <Sycl_Graph::Sycl::Graph_type Graph_t,
            Sycl_Graph::Property_Extractor_type... Es>
  std::tuple<sycl::buffer<typename Es::Apply_t>...> construct_apply_buffers(
      Graph_t& graph, const std::tuple<Es...>& extractors) {
    auto edge_sizes = std::apply(
        [&](auto... extractor) {
          return std::make_tuple(graph.edge_buf.template current_size<typename Es::Edge_t>()...);
        },
        extractors);

    std::tuple<sycl::buffer<typename Es::Apply_t>...> bufs = std::apply(
        [&](auto&... edge_size) {
          return std::make_tuple(sycl::buffer<typename Es::Apply_t>(edge_size)...);
        },
        edge_sizes);

    return bufs;
  }

  template <Sycl_Graph::Sycl::Graph_type Graph_t,
            Sycl_Graph::Property_Extractor_type... Es>
  std::tuple<sycl::buffer<typename Es::Accumulate_t>...> construct_accumulation_buffers(
      Graph_t& graph, const std::tuple<Es...>& extractors) {
    auto N_vertices = graph.N_vertices();
    auto edge_sizes = std::apply(
        [&](auto... extractor) {
          return std::make_tuple(graph.edge_buf.template current_size<typename Es::Edge_t>()...);
        },
        extractors);
    std::tuple<sycl::buffer<typename Es::Accumulate_t>...> bufs = std::make_tuple(sycl::buffer<typename Es::Accumulate_t>(N_vertices)...);
    return bufs;
  }

  template <Sycl_Graph::Sycl::Graph_type Graph_t,
            Sycl_Graph::Property_Extractor_type... Es>
  std::tuple<std::vector<typename Es::Accumulate_t>...> extract_properties(
      Graph_t& graph, const std::tuple<Es...>& extractors, sycl::queue& q) {
    std::tuple<sycl::buffer<typename Es::Apply_t>...> apply_buffers
        = construct_apply_buffers(graph, extractors);
    auto accumulate_buffers = construct_accumulation_buffers(graph, extractors);

    auto apply_events = extractor_apply(graph, extractors, apply_buffers, q);
    auto accumulate_events = extractor_accumulate(graph, extractors, apply_buffers,
                                                  accumulate_buffers, q, apply_events);
    q.wait();

    std::tuple<std::vector<typename Es::Accumulate_t>...> results;
    auto buffer_cpy = [&](auto& res, auto& buf) { res = buffer_get(buf); };
    std::apply(
        [&](auto&... result) {
          return std::apply(
              [&](auto&... accumulate_buffer) { ((buffer_cpy(result, accumulate_buffer), ...)); },
              accumulate_buffers);
        },
        results);

    return results;
  }

}  // namespace Sycl_Graph::Sycl
#endif