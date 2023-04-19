#ifndef SYCL_GRAPH_ALGORITHMS_PROPERTIES_SYCL_DEGREE_PROPERTIES_HPP
#define SYCL_GRAPH_ALGORITHMS_PROPERTIES_SYCL_DEGREE_PROPERTIES_HPP
#include <CL/sycl.hpp>
#include <Sycl_Graph/Algorithms/Properties/Sycl/Property_Extractor.hpp>
#include <Sycl_Graph/Buffer/Invariant/Buffer.hpp>
namespace Sycl_Graph::Sycl {
  enum Degree_Property { In_Degree, Out_Degree };

  template <Sycl_Graph::Base::Vertex_Buffer_type _Vertex_Buffer_From_t,
            Sycl_Graph::Base::Vertex_Buffer_type _Vertex_Buffer_To_t,
            Sycl_Graph::Base::Edge_Buffer_type _Edge_Buffer_t, typename _Apply_t,
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

    virtual Apply_t apply(const Edge_t& edge, const From_t& data_from, const To_t& data_to) const = 0;
    virtual Accumulate_t accumulate(
        const sycl::accessor<Apply_t, 1, sycl::access::mode::read>& apply_acc,
        const sycl::accessor<Accumulate_t, 1, sycl::access::mode::read_write>& accumulate_acc,
        uint32_t idx) const
        = 0;
  };

  template <Sycl_Graph::Base::Vertex_Buffer_type From, Sycl_Graph::Base::Vertex_Buffer_type To>
  struct Degree_Apply_t {
    typename From::ID_t from;
    typename To::ID_t to;
  };

  template <Sycl_Graph::Base::Vertex_Buffer_type From, Sycl_Graph::Base::Vertex_Buffer_type To>
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

  template <Sycl_Graph::Base::Vertex_Buffer_type _Vertex_Buffer_From_t,
            Sycl_Graph::Base::Vertex_Buffer_type _Vertex_Buffer_To_t,
            Sycl_Graph::Base::Edge_Buffer_type _Edge_Buffer_t>
  struct Degree_Extractor : public Undirected_Extractor<
                                _Vertex_Buffer_From_t, _Vertex_Buffer_To_t, _Edge_Buffer_t,
                                Degree_Apply_t<_Vertex_Buffer_From_t, _Vertex_Buffer_To_t>,
                                Degree_Accumulate_t<_Vertex_Buffer_From_t, _Vertex_Buffer_To_t>> {
    using Base_t
        = Undirected_Extractor<_Vertex_Buffer_From_t, _Vertex_Buffer_To_t, _Edge_Buffer_t,
                               Degree_Apply_t<_Vertex_Buffer_From_t, _Vertex_Buffer_To_t>,
                               Degree_Accumulate_t<_Vertex_Buffer_From_t, _Vertex_Buffer_To_t>>;
    using Base_t::Base_t;
    using typename Base_t::Accumulate_t;
    using typename Base_t::Apply_t;
    using typename Base_t::Edge_t;
    using typename Base_t::From_t;
    using typename Base_t::To_t;

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

  template <Sycl_Graph::Base::Vertex_Buffer_type _Vertex_Buffer_From_t,
            Sycl_Graph::Base::Vertex_Buffer_type _Vertex_Buffer_To_t,
            Sycl_Graph::Base::Edge_Buffer_type _Edge_Buffer_t>
  Degree_Extractor(const _Vertex_Buffer_From_t&,
                                     const _Vertex_Buffer_To_t&,
                                     const _Edge_Buffer_t&)
      ->Degree_Extractor<_Vertex_Buffer_From_t, _Vertex_Buffer_To_t, _Edge_Buffer_t>;

  //   template <Sycl_Graph::Base::Vertex_Buffer_type _Vertex_Buffer_t,
  //             Sycl_Graph::Base::Edge_Buffer_type _Edge_Buffer_t>
  //   struct Degree_Square_Sum_Extractor {
  //     typedef _Vertex_Buffer_t Vertex_Buffer_t;
  //     typedef _Edge_Buffer_t Edge_Buffer_t;
  //     typedef typename Edge_Buffer_t::Edge_t Edge_t;
  //     typedef typename Vertex_Buffer_t::uI_t uI_t;
  //     typedef typename Vertex_Buffer_t::ID_t ID_t;
  //     typedef typename Edge_Buffer_t::Edge_t::From_t From_t;
  //     typedef typename Edge_Buffer_t::Edge_t::To_t To_t;

  //     typedef ID_t Apply_t;
  //     typedef sycl::accessor<Apply_t, 1, sycl::access::mode::read> Apply_Access_t;
  //     typedef std::pair<ID_t, uI_t> Accumulation_Apply_t;
  //     typedef sycl::accessor<Accumulation_Apply_t, 1, sycl::access::mode::write>
  //     Accumulate_Access_t; Degree_Property property;

  //     Degree_Square_Sum_Extractor(const Vertex_Buffer_t& vertex_buffer,
  //                                 const Edge_Buffer_t& edge_buffer,
  //                                 Degree_Property property = In_Degree)
  //         : property(property) {}

  //     Apply_t apply(const Edge_t& edge_target) {
  //       if constexpr (property == In_Degree) {
  //         return edge_target.to;
  //       } else if constexpr (property == Out_Degree) {
  //         return edge_target.from;
  //       }
  //     }

  //     void accumulate(const Accumulation_Apply_t& return_properties,
  //                     Accumulate_Access_t& accumulated_property) {
  //       for (int i = 0; i < return_properties.size(); i++) {
  //         for (int j = 0; j < accumulated_property.size(); j++) {
  //           if (return_properties[i] == accumulated_property[j].first) {
  //             accumulated_property[j].second++;
  //           }
  //         }
  //       }

  //       for (int j = 0; j < accumulated_property.size(); j++) {
  //         accumulated_property[j].second
  //             = accumulated_property[j].second * accumulated_property[j].second;
  //       }
  //     }
  //   };
}  // namespace Sycl_Graph::Sycl
#endif