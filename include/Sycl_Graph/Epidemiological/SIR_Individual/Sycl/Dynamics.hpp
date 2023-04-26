#ifndef SYCL_GRAPH_EPIDEMIOLOGICAL_SIR_INDIVIDUAL_DYNAMICS_HPP
#define SYCL_GRAPH_EPIDEMIOLOGICAL_SIR_INDIVIDUAL_DYNAMICS_HPP
#include <Sycl_Graph/Epidemiological/SIR/Types.hpp>
#include <Sycl


namespace Sycl_Graph::Epidemiological
{

typedef 


template <Sycl_Graph::Vertex_Buffer_type _Vertex_Buffer_t>
  struct Recovery_Extractor
  {
    typedef uint32_t Accumulate_t;
    typedef SIR_Individual_State_t Apply_t;
    typedef typename _Vertex_Buffer_t::Vertex_t Vertex_t;
    Recovery_Extractor() = default;
    Recovery_Extractor(const _Vertex_Buffer& vertex_buffer) {}


    Apply_t apply(const Edge_t& edge, const From_t& from, const To_t& to) const {

    }

    void synchronize(
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
}


#endif
