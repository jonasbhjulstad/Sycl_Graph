#ifndef SYCL_GRAPH_EPIDEMIOLOGICAL_SIR_INDIVIDUAL_OPERATIONS_HPP
#define SYCL_GRAPH_EPIDEMIOLOGICAL_SIR_INDIVIDUAL_OPERATIONS_HPP
#include <Static_RNG/distributions.hpp>
#include <Sycl_Graph/Epidemiological/SIR_Individual/Types.hpp>
#include <Sycl_Graph/Graph/Base/Graph_Types.hpp>
#include <random>
namespace Sycl_Graph::Epidemiological {
  template <Sycl_Graph::Vertex_Buffer_type Vertex_Buffer_t> struct SIR_Vertex_Recovery_Op {
    typedef SIR_Individual_State_t Result_t;

    static constexpr Operation_Target_t operation_target = Operation_Target_Vertex;
    static constexpr Operation_Type_t operation_type = Operation_Direct_Transform;
    static constexpr sycl::access::mode vertex_access_mode = sycl::access::mode::read;
    typedef typename Vertex_Buffer_t::Vertex_t Vertex_t;
    sycl::buffer<uint32_t> seeds;
    float p_R = 0.0f;
    SIR_Vertex_Recovery_Sample_Op(const std::tuple<Vertex_Buffer_t>& buf, sycl::queue& q,
                                  float p_R = 0.0f, uint32_t seed = 0)
        : seeds(N_threads), p_R(p_R) {
      seeds = generate_seed_buf(seed, buf.current_size(), q);
    }

    void operator()(const auto& v_acc, auto& result_acc, sycl::handler& h) const {
      if (direction == Degree_Direction_From) {
        auto seed_acc = seeds.template get_access<sycl::access::mode::read_write>(h);
        h.parallel_for(from_acc.size(), [=](sycl::id<1> id) {
          result_acc[id] = false;
          auto seed = seed_acc[id];
          seed_acc[id] += 1;
          Static_RNG::default_rng rng(seed);
          Static_RNG::bernoulli_distribution<float> dist(p_R);
          result_acc[id] = dist(rng) ? SIR_Individual_R : v_acc[id];
        });
      }
    }
    template <Graph_type Graph_t> size_t result_buffer_size(const Graph_t& G) const {
      return G.vertex_buf.template get_buffer<Vertex_Buffer_t>().current_size();
    }
  };

  struct SIR_Individual_Infection_Op {
    static constexpr Operation_Target_t operation_target = Operation_Target_Edge;
    static constexpr Operation_Type_t operation_type = Operation_Buffer_Transform;
    static constexpr Edge_Direction_t direction = bidirectional;
    static constexpr sycl::access::mode result_access_mode = sycl::access::mode::atomic;
    typedef typename SIR_Individual_Edge_Buffer_t::Edge_t Edge_t;
    typedef typename SIR_Individual_Vertex_Buffer_t::Vertex_t From_t;
    typedef typename SIR_Individual_Vertex_Buffer_t::Vertex_t To_t;
    sycl::buffer<uint32_t> seeds;
    float p_I = 0.0f;
    SIR_Individual_Infection_Events_Op(const Vertex_Buffer_From_t&, const Vertex_Buffer_To_t&,
                                       const Edge_Buffer_t& edge_buf, sycl::queue& q,
                                       float p_I = 0.0f, uint32_t N_threads = 1024,
                                       uint32_t seed = 0)
        : seeds(N_threads), p_I(p_I) {
      seeds = generate_seed_buf(seed, N_threads, q);
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
    template <Graph_type Graph_t> size_t result_buffer_size(const Graph_t& G) const {
      return G.vertex_buf.template get_buffer<Vertex_Buffer_From_t>().current_size();
      else return G.vertex_buf.template get_buffer<Vertex_Buffer_To_t>().current_size();
    }
  };

}  // namespace Sycl_Graph::Epidemiological

#endif