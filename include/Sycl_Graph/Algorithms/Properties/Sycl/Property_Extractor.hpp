#ifndef SYCL_GRAPH_ALGORITHMS_PROPERTIES_SYCL_PROPERTY_EXTRACTOR_HPP
#define SYCL_GRAPH_ALGORITHMS_PROPERTIES_SYCL_PROPERTY_EXTRACTOR_HPP

#include <CL/sycl.hpp>
#include <Sycl_Graph/Algorithms/Properties/Invariant/Property_Extractor.hpp>
#include <Sycl_Graph/Buffer/Sycl/type_helpers.hpp>
#include <tuple>

namespace Sycl_Graph::Sycl {
  template <typename T>
  concept Property_Extractor_type = Sycl_Graph::Invariant::Property_Extractor_type<T>;

  template <Sycl_Graph::Invariant::Graph_type Graph_t, Property_Extractor_type... Es>
  sycl::event single_type_extractor_apply(
      Graph_t& graph, const std::tuple<Es...>& extractors,
      std::tuple<sycl::buffer<typename Es::Property_t>...>& apply_bufs, sycl::queue& q) {
    using Edge_t = typename std::tuple_element<0, std::tuple<Es...>>::type::Edge_t;

    return q.submit([&](sycl::handler& h) {
      auto apply_acc = std::apply(
          [&](auto&... apply_buf) {
            return std::make_tuple(apply_buf.template get_access<sycl::access::mode::read>(h)...);
          },
          apply_bufs);
      auto edge_acc = graph.edge_buf.template get_access<sycl::access::mode::read, Edge_t>(h);
      auto to_acc
          = graph.vertex_buf.template get_access<sycl::access::mode::read, typename Edge_t::To_t>(
              h);
      auto from_acc
          = graph.vertex_buf.template get_access<sycl::access::mode::read, typename Edge_t::From_t>(
              h);
      h.parallel_for(graph.edge_buf.get_count(), [=](sycl::id<1> i) {
        auto edge = edge_acc[i];
        auto to = to_acc[edge.to];
        auto from = from_acc[edge.from];
        std::apply(
            [&](auto... ex) {
              std::apply([&](auto&... ap) { ((apply_acc[i] = ex.apply(edge, from, to)), ...); },
                         apply_acc);
            },
            extractors);
      });
    });
  }

  template <Sycl_Graph::Invariant::Graph_type Graph_t, Property_Extractor_type... Es>
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



  template <Sycl_Graph::Invariant::Graph_type Graph_t, Property_Extractor_type... Es>
  auto extractor_apply(
      Graph_t& graph, const std::tuple<Es...>& extractors,
      std::tuple<sycl::buffer<typename Es::Property_t> ...>& apply_buf,
      std::tuple<sycl::buffer<typename Es::Accumulation_Property_t> ...>& accumulate_buf,
      sycl::queue& q) {
    auto extractor_list = separate_by_edge_type(extractors);
    auto apply_buf_list = separate_by_type(apply_buf);
    auto apply_events = std::apply(
        [&](auto... apply_bufs) {
          return std::apply(
              [&](auto... extractors) {
                return std::make_tuple(
                    (single_type_extractor_apply(graph, extractors, apply_bufs, q), ...));
              },
              extractor_list);
        },
        apply_buf_list);
    return apply_events;
  }

  template <Sycl_Graph::Invariant::Graph_type Graph_t, Property_Extractor_type... Es>
  auto extractor_accumulate(
      Graph_t& graph, const std::tuple<Es...>& extractors,
      std::tuple<sycl::buffer<typename Es::Property_t> ...>& apply_buf,
      std::tuple<sycl::buffer<typename Es::Accumulation_Property_t> ...>& accumulate_buf,
      sycl::queue& q, auto& apply_events) {
    auto extractor_list = separate_by_edge_type(extractors);
    auto apply_buf_list = separate_by_type(apply_buf);
    auto accumulate_events = std::apply(
        [&](auto... apply_event) {
          std::apply(
              [&](auto... apply_bufs) {
                return std::apply(
                    [&](auto... extractors) {
                      return std::make_tuple((
                          single_type_extractor_accumulate(graph, extractors, apply_bufs, q), ...));
                    },
                    extractor_list);
              },
              apply_buf_list);
        },
        apply_events);
    return accumulate_events;
  }
  template <Sycl_Graph::Invariant::Graph_type Graph_t, Property_Extractor_type... Es>
  std::tuple<sycl::buffer<typename Es::Property_t>...> construct_apply_buffers(
      Graph_t& graph, const std::tuple<Es...>& extractors) {
    auto edge_sizes = std::apply(
        [&](auto... extractor) {
          return std::make_tuple(graph.template current_size<typename Es::Edge_t>()...);
        },
        extractors);


    std::tuple<sycl::buffer<typename Es::Property_t>...> bufs = std::apply(
        [&](auto&... edge_size) {
          return std::make_tuple(
              sycl::buffer<typename Es::Property_t>(edge_size)...);
        },
        edge_sizes);

    return bufs;
  }

  template <Sycl_Graph::Invariant::Graph_type Graph_t, Property_Extractor_type... Es>
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


  template <Sycl_Graph::Invariant::Graph_type Graph_t, Property_Extractor_type... Es>
  std::tuple<std::vector<typename Es::Accumulation_Property_t>...> extract_properties(
      Graph_t& graph, const std::tuple<Es...>& extractors, sycl::queue& q) {
    std::tuple<sycl::buffer<typename Es::Property_t> ...> apply_buffers = construct_apply_buffers(graph, extractors);
    auto accumulate_buffers = construct_accumulation_buffers(graph, extractors);

    
    auto apply_events = extractor_apply(graph, extractors, apply_buffers, accumulate_buffers, q);
    auto accumulate_events = extractor_accumulate(graph, extractors, apply_buffers,
                                                  accumulate_buffers, q, apply_events);
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