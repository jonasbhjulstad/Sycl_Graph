#ifndef SYCL_GRAPH_EPIDEMIOLOGICAL_SIR_INDIVIDUAL_OPERATIONS_HPP
#define SYCL_GRAPH_EPIDEMIOLOGICAL_SIR_INDIVIDUAL_OPERATIONS_HPP
#include <Static_RNG/distributions.hpp>
#include <Sycl_Graph/Epidemiological/SIR_Individual/Types.hpp>
#include <Sycl_Graph/Graph/Base/Graph_Types.hpp>
#include <random>
namespace Sycl_Graph::Epidemiological {
  template <Sycl_Graph::Vertex_Buffer_type Vertex_Buffer_t> struct SIR_Vertex_Recovery_Op {
    typedef SIR_Individual_State_t Result_t;
    typedef typename Vertex_Buffer_t::Vertex_t Vertex_t;

    typedef Vertex_t Iterator_t;
    typedef SIR_Individual_State_t Target_t;
    static constexpr sycl::access::mode target_access_mode = sycl::access::mode::write;

    sycl::buffer<uint32_t>& seeds;
    float p_R = 0.0f;
    SIR_Vertex_Recovery_Sample_Op(const Vertex_Buffer_t& buf, sycl::buffer<uint32_t>& seeds,
                                  float p_R = 0.0f)
        : p_R(p_R), seeds(seeds) {
      assert(seeds.size() >= buf.current_size() && "seeds buffer too small");
    }

    void operator()(const auto& v_acc, auto& result_acc, sycl::handler& h) const {
      if (direction == Degree_Direction_From) {
        auto seed_acc = seeds.template get_access<sycl::access::mode::read_write>(h);
        h.parallel_for(v_acc.size(), [=](sycl::id<1> id) {
          result_acc[id] = false;
          auto seed = seed_acc[id];
          seed_acc[id] += 1;
          Static_RNG::default_rng rng(seed);
          Static_RNG::bernoulli_distribution<float> dist(p_R);
          result_acc[id] = dist(rng) ? SIR_Individual_R : v_acc[id];
        });
      }
    }
    template <Graph_type Graph_t> size_t target_buffer_size(const Graph_t& G) const {
      return G.vertex_buf.template get_buffer<Vertex_Buffer_t>().current_size();
    }
  };

  struct SIR_Individual_Infection_Op {
    typedef typename SIR_Individual_Edge_Buffer_t::Edge_t Edge_t;
    typedef SIR_Individual_State_t Inplace_t;
    typedef Iterator_t Edge_t;
    static constexpr sycl::access::mode target_access_mode = sycl::access::mode::atomic;
    typedef typename SIR_Individual_Vertex_Buffer_t::Vertex_t From_t;
    typedef typename SIR_Individual_Vertex_Buffer_t::Vertex_t To_t;
    sycl::buffer<uint32_t>& seeds;
    float p_I = 0.0f;
    SIR_Individual_Infection_Op(const Vertex_Buffer_From_t&, const Vertex_Buffer_To_t&,
                                       const Edge_Buffer_t& edge_buf, sycl::buffer<uint32_t>& seeds,
                                       float p_I = 0.0f)
        : seeds(seeds), p_I(p_I) {
      assert(seeds.size() >= edge_buf.current_size() && "seeds buffer too small");
    }

    void operator()(const auto& edge_acc, const auto& from_acc, const auto& to_acc,
                    auto& result_acc, sycl::handler& h) const {
      h.parallel_for(edge_acc.size(), [=](sycl::id<1> id) {
        auto id_from = edge_acc[id].from;
        auto id_to = edge_acc[id].to;
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
  };

}  // namespace Sycl_Graph::Epidemiological

#endif