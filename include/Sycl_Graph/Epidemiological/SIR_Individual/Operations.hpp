#ifndef SYCL_GRAPH_EPIDEMIOLOGICAL_SIR_INDIVIDUAL_OPERATIONS_HPP
#define SYCL_GRAPH_EPIDEMIOLOGICAL_SIR_INDIVIDUAL_OPERATIONS_HPP
#include <Static_RNG/distributions.hpp>
#include <Sycl_Graph/Algorithms/Operations/Operations.hpp>
#include <Sycl_Graph/Epidemiological/SIR_Individual/Types.hpp>
#include <Sycl_Graph/Graph/Base/Graph_Types.hpp>
#include <random>
namespace Sycl_Graph::Epidemiological {
  using namespace Sycl_Graph::Sycl;


  template <Sycl_Graph::Vertex_Buffer_type Vertex_Buffer_t> struct SIR_Vertex_Recovery_Op
      : public Vertex_Extract_Operation<Vertex_Buffer_t, SIR_Vertex_Recovery_Op<Vertex_Buffer_t>> {
    typedef SIR_Individual_State_t Target_t;
    typedef typename Vertex_Buffer_t::Vertex_t Vertex_t;

    typedef SIR_Individual_State_t Target_t;

    float p_R = 0.0f;
    SIR_Vertex_Recovery_Op(const Vertex_Buffer_t& buf, float p_R = 0.0f) : p_R(p_R) {}

    void invoke(const auto& v_acc, auto& result_acc, sycl::handler& h) const {
      if (direction == Degree_Direction_From) {
        h.parallel_for(v_acc.size(), [=](sycl::id<1> id) {
          Static_RNG::default_rng rng(id);
          Static_RNG::bernoulli_distribution<float> dist(p_R);
          result_acc[id] = dist(rng) ? SIR_Individual_R : v_acc[id];
        });
      }
    }
    template <Graph_type Graph_t> size_t target_buffer_size(const Graph_t& G) const {
      return G.vertex_buf.template get_buffer<Vertex_Buffer_t>().current_size();
    }
  };


  //Individual Infection Op: Chained with Individual Recovery Op as an inplace operation
  struct SIR_Individual_Infection_Op
      : public Edge_Transform_Operation<SIR_Individual_Infection_Op> {
    using Base_t = Edge_Transform_Operation<SIR_Individual_Infection_Op>;
    static constexpr sycl::access::mode target_access_mode = sycl::access::mode::atomic;
    typedef SIR_Individual_State_t Target_t;
    typedef SIR_Individual_State_t Source_t;
    sycl::buffer<uint32_t>& seeds;
    float p_I = 0.0f;
    SIR_Individual_Infection_Op(const Vertex_Buffer_From_t&, const Vertex_Buffer_To_t&,
                                const Edge_Buffer_t& edge_buf, sycl::buffer<uint32_t>& seeds,
                                float p_I = 0.0f)
        : seeds(seeds), p_I(p_I) {
      assert(seeds.size() >= edge_buf.current_size() && "seeds buffer too small");
    }

    void invoke(const auto& edge_acc, const auto& from_acc, const auto& to_acc,
                    auto& result_acc, sycl::handler& h) const {
      h.parallel_for(edge_acc.size(), [=](sycl::id<1> id) {
        auto id_from = edge_acc[id].from;
        auto id_to = edge_acc[id].to;
        auto seed_acc = seeds.get_access<sycl::access::mode::read_write>();
        if (edge_acc[id].is_valid() && (from_acc[id_from] == SIR_INDIVIDUAL_I)
            && (to_acc[id_to] == SIR_INDIVIDUAL_S)) {
          Static_RNG::default_rng rng(seed_acc[id]);
          seed_acc[id]++;
          auto p_I = edge_acc[id].data.p_I;
          Static_RNG::bernoulli_distribution<float> dist(p_I);
          if (dist(rng)) {
            to_acc[id_to] = SIR_INDIVIDUAL_I;
          }
          seed_acc[id]++;
        }
      });
    }
    template <Graph_type Graph_t> size_t target_buffer_size(const Graph_t& G) const {
      return G.vertex_buf.template get_buffer<Vertex_Buffer_To_t>().current_size();
    }

    template <Graph_type Graph_t> size_t source_buffer_size(const Graph_t& G) const {
      return G.vertex_buf.template get_buffer<Vertex_Buffer_To_t>().current_size();
    }
  };

}  // namespace Sycl_Graph::Epidemiological

#endif