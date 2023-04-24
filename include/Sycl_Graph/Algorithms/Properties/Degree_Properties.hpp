#ifndef SYCL_GRAPH_ALGORITHMS_PROPERTIES_SYCL_DEGREE_PROPERTIES_HPP
#define SYCL_GRAPH_ALGORITHMS_PROPERTIES_SYCL_DEGREE_PROPERTIES_HPP
#include <concepts>
#include <CL/sycl.hpp>
#include <Sycl_Graph/Algorithms/Properties/Property_Extractor.hpp>
#include <Sycl_Graph/Buffer/Base/Buffer_Pack.hpp>
namespace Sycl_Graph::Sycl {
  enum Degree_Property { In_Degree, Out_Degree };

  template <Sycl_Graph::Vertex_Buffer_type From, Sycl_Graph::Vertex_Buffer_type To>
  struct Directed_Degree_Apply_t {
    typename From::ID_t from;
    typename To::ID_t to;
  };

  template <Sycl_Graph::Vertex_Buffer_type From, Sycl_Graph::Vertex_Buffer_type To>
  struct Directed_Degree_Accumulate_t {
    uint32_t from = 0;
    uint32_t to = 0;
  };

  template <Sycl_Graph::Vertex_Buffer_type _Vertex_Buffer_From_t,
            Sycl_Graph::Vertex_Buffer_type _Vertex_Buffer_To_t,
            Sycl_Graph::Edge_Buffer_type _Edge_Buffer_t>
  struct Directed_Degree_Extractor
  {
    typedef Directed_Degree_Accumulate_t<_Vertex_Buffer_From_t, _Vertex_Buffer_To_t> Accumulate_t;
    typedef Directed_Degree_Apply_t<_Vertex_Buffer_From_t, _Vertex_Buffer_To_t> Apply_t;
    typedef typename _Edge_Buffer_t::Edge_t Edge_t;
    typedef typename _Vertex_Buffer_From_t::Vertex_t From_t;
    typedef typename _Vertex_Buffer_To_t::Vertex_t To_t;

    Directed_Degree_Extractor() = default;
    Directed_Degree_Extractor(const _Vertex_Buffer_From_t& vertex_buffer_from,
                     const _Vertex_Buffer_To_t& vertex_buffer_to,
                     const _Edge_Buffer_t& edge_buffer) {}


    Apply_t apply(const Edge_t& edge, const From_t& from, const To_t& to) const {
      return {edge.from, edge.to};
    }

    void accumulate(
        const sycl::accessor<Apply_t, 1, sycl::access::mode::read>& apply_acc,
        const sycl::accessor<Accumulate_t, 1, sycl::access::mode::read_write>& accumulate_acc,
        uint32_t idx) const {
      accumulate_acc[idx] = Accumulate_t();
      for (int i = 0; i < apply_acc.size(); i++) {
        if (apply_acc[i].from == idx) {
          accumulate_acc[idx].from++;
        }
        if (apply_acc[i].to == idx) {
          accumulate_acc[idx].to++;
        }
      }
    }
  };


  template <Sycl_Graph::Vertex_Buffer_type From, Sycl_Graph::Vertex_Buffer_type To>
  struct Undirected_Degree_Apply_t {
    typename From::ID_t from;
    typename To::ID_t to;
  };


  template <Sycl_Graph::Vertex_Buffer_type _Vertex_Buffer_From_t,
            Sycl_Graph::Vertex_Buffer_type _Vertex_Buffer_To_t,
            Sycl_Graph::Edge_Buffer_type _Edge_Buffer_t>
  struct Undirected_Degree_Extractor
  {
    typedef uint32_t Accumulate_t;
    typedef Directed_Degree_Apply_t<_Vertex_Buffer_From_t, _Vertex_Buffer_To_t> Apply_t;
    typedef typename _Edge_Buffer_t::Edge_t Edge_t;
    typedef typename _Vertex_Buffer_From_t::Vertex_t From_t;
    typedef typename _Vertex_Buffer_To_t::Vertex_t To_t;

    Undirected_Degree_Extractor() = default;
    Undirected_Degree_Extractor(const _Vertex_Buffer_From_t& vertex_buffer_from,
                     const _Vertex_Buffer_To_t& vertex_buffer_to,
                     const _Edge_Buffer_t& edge_buffer) {}


    Apply_t apply(const Edge_t& edge, const From_t& from, const To_t& to) const {
      return {edge.from, edge.to};
    }

    void accumulate(
        const sycl::accessor<Apply_t, 1, sycl::access::mode::read>& apply_acc,
        const sycl::accessor<Accumulate_t, 1, sycl::access::mode::read_write>& accumulate_acc,
        uint32_t idx) const {
      accumulate_acc[idx] = 0;
      for (int i = 0; i < apply_acc.size(); i++) {
        if ((apply_acc[i].from == idx) || (apply_acc[i].to == idx)) {
          accumulate_acc[idx]++;
      }
      }
    }   
  };

  template <Sycl_Graph::Vertex_Buffer_type _Vertex_Buffer_From_t,
            Sycl_Graph::Vertex_Buffer_type _Vertex_Buffer_To_t,
            Sycl_Graph::Edge_Buffer_type _Edge_Buffer_t>
  Directed_Degree_Extractor(const _Vertex_Buffer_From_t&,
                                     const _Vertex_Buffer_To_t&,
                                     const _Edge_Buffer_t&)
      ->Directed_Degree_Extractor<_Vertex_Buffer_From_t, _Vertex_Buffer_To_t, _Edge_Buffer_t>;

  
  template <Sycl_Graph::Vertex_Buffer_type _Vertex_Buffer_From_t,
            Sycl_Graph::Vertex_Buffer_type _Vertex_Buffer_To_t,
            Sycl_Graph::Edge_Buffer_type _Edge_Buffer_t>
  Undirected_Degree_Extractor(const _Vertex_Buffer_From_t&,
                                     const _Vertex_Buffer_To_t&,
                                     const _Edge_Buffer_t&)
      ->Undirected_Degree_Extractor<_Vertex_Buffer_From_t, _Vertex_Buffer_To_t, _Edge_Buffer_t>;

}  // namespace Sycl_Graph::Sycl
#endif