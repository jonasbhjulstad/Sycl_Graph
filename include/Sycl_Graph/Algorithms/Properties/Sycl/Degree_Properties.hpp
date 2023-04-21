#ifndef SYCL_GRAPH_ALGORITHMS_PROPERTIES_SYCL_DEGREE_PROPERTIES_HPP
#define SYCL_GRAPH_ALGORITHMS_PROPERTIES_SYCL_DEGREE_PROPERTIES_HPP
#include <concepts>
#include <CL/sycl.hpp>
#include <Sycl_Graph/Algorithms/Properties/Sycl/Property_Extractor.hpp>
#include <Sycl_Graph/Buffer/Base/Buffer_Pack.hpp>
namespace Sycl_Graph::Sycl {
  enum Degree_Property { In_Degree, Out_Degree };

  template <Sycl_Graph::Vertex_Buffer_type _Vertex_Buffer_From_t,
            Sycl_Graph::Vertex_Buffer_type _Vertex_Buffer_To_t,
            Sycl_Graph::Edge_Buffer_type _Edge_Buffer_t, typename _Apply_t,
            typename _Accumulate_t>
  struct Undirected_Extractor {
    typedef _Vertex_Buffer_To_t Vertex_Buffer_To_t;
    typedef _Vertex_Buffer_From_t Vertex_Buffer_From_t;
    typedef _Edge_Buffer_t Edge_Buffer_t;
    typedef typename Edge_Buffer_t::Edge_t Edge_t;
    typedef typename Edge_t::From_t From_t;
    typedef typename Edge_t::To_t To_t;
    typedef _Apply_t Apply_t;
    typedef _Accumulate_t Accumulate_t;

    Undirected_Extractor() = default;
    Undirected_Extractor(const Vertex_Buffer_From_t& vertex_buffer_from,
                         const Vertex_Buffer_To_t& vertex_buffer_to,
                         const Edge_Buffer_t& edge_buffer){}

    Apply_t apply(const Edge_t& edge, const From_t& data_from, const To_t& data_to) const = 0;
    Accumulate_t accumulate(
        const sycl::accessor<Apply_t, 1, sycl::access::mode::read>& apply_acc,
        const sycl::accessor<Accumulate_t, 1, sycl::access::mode::read_write>& accumulate_acc,
        uint32_t idx) const
        {
        }
  };


  template <typename T>
  concept Undirected_Extractor_type = 
  std::is_member_function_pointer_v<decltype(&T::accumulate)> &&
  requires(typename T::Edge_t edge, typename T::From_t data_from, typename T::To_t data_to)
            {
              T::Vertex_Buffer_To_t;
              T::Vertex_Buffer_From_t;
              T::Edge_Buffer_t;
              T::Apply_t;
              T::Accumulate_t;
              T::apply(edge, data_from, data_to);
            };

  template <Sycl_Graph::Vertex_Buffer_type From, Sycl_Graph::Vertex_Buffer_type To>
  struct Degree_Apply_t {
    typename From::ID_t from;
    typename To::ID_t to;
  };

  template <Sycl_Graph::Vertex_Buffer_type From, Sycl_Graph::Vertex_Buffer_type To>
  struct Degree_Accumulate_t {
    struct {
      typename From::ID_t id;
      uint32_t degree;
    } from;
    struct {
      typename To::ID_t id;
      uint32_t degree;
    } to;
  };

  template <Sycl_Graph::Vertex_Buffer_type _Vertex_Buffer_From_t,
            Sycl_Graph::Vertex_Buffer_type _Vertex_Buffer_To_t,
            Sycl_Graph::Edge_Buffer_type _Edge_Buffer_t>
  struct Degree_Extractor
  {
    typedef Degree_Accumulate_t<_Vertex_Buffer_From_t, _Vertex_Buffer_To_t> Accumulate_t;
    typedef Degree_Apply_t<_Vertex_Buffer_From_t, _Vertex_Buffer_To_t> Apply_t;
    typedef typename _Edge_Buffer_t::Edge_t Edge_t;
    typedef typename _Vertex_Buffer_From_t::Data_t From_t;
    typedef typename _Vertex_Buffer_To_t::Data_t To_t;

    Degree_Extractor() = default;
    Degree_Extractor(const _Vertex_Buffer_From_t& vertex_buffer_from,
                     const _Vertex_Buffer_To_t& vertex_buffer_to,
                     const _Edge_Buffer_t& edge_buffer) {}


    Apply_t apply(const Edge_t& edge, const From_t& from, const To_t& to) const {
      return {edge.from, edge.to};
    }

    Accumulate_t accumulate(
        const sycl::accessor<Apply_t, 1, sycl::access::mode::read>& apply_acc,
        const sycl::accessor<Accumulate_t, 1, sycl::access::mode::read_write>& accumulate_acc,
        uint32_t idx) const {
      for (int i = 0; i < apply_acc.size(); i++) {
        if (apply_acc[i].from == accumulate_acc[idx].from.id) {
          accumulate_acc[idx].from.degree++;
        }
        if (apply_acc[i].to == accumulate_acc[idx].to.id) {
          accumulate_acc[idx].to.degree++;
        }
      }
    }
  };

  template <Sycl_Graph::Vertex_Buffer_type _Vertex_Buffer_From_t,
            Sycl_Graph::Vertex_Buffer_type _Vertex_Buffer_To_t,
            Sycl_Graph::Edge_Buffer_type _Edge_Buffer_t>
  Degree_Extractor(const _Vertex_Buffer_From_t&,
                                     const _Vertex_Buffer_To_t&,
                                     const _Edge_Buffer_t&)
      ->Degree_Extractor<_Vertex_Buffer_From_t, _Vertex_Buffer_To_t, _Edge_Buffer_t>;

}  // namespace Sycl_Graph::Sycl
#endif