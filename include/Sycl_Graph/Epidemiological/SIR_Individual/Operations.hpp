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
    struct Result_t {
      bool from = false;
      bool to = false;
    };

    static constexpr Operation_Target_t operation_target = Operation_Target_Edge;
    static constexpr Operation_Type_t operation_type = Operation_Buffer_Transform;
    Degree_Direction direction = Degree_Direction_From;
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
      h.parallel_for(from_acc.size(), [=](sycl::id<1> id) {
        result_acc[id] = Result_t();
        if (from_acc[id] == SIR_Individual_I && to_acc[id] == SIR_Individual_S)

          result_acc[id].from = true;
      });
    }
    template <Graph_type Graph_t> size_t result_buffer_size(const Graph_t& G) const {
        return G.vertex_buf.template get_buffer<Vertex_Buffer_From_t>().current_size();
      else
        return G.vertex_buf.template get_buffer<Vertex_Buffer_To_t>().current_size();
    }
  };

  template <Sycl_Graph::Vertex_Buffer_type Vertex_Buffer_t> struct SIR_Vertex_Recovery_Op {
    typedef SIR_Individual_State_t Result_t;

    static constexpr Operation_Target_t operation_target = Operation_Target_Vertex;
    static constexpr Operation_Type_t operation_type = Operation_Direct_Transform;
    typedef typename Vertex_Buffer_t::Vertex_t Vertex_t;
    sycl::buffer<uint32_t> seeds;
    float p_R = 0.0f;
    SIR_Vertex_Recovery_Modify_Op(const std::tuple<Vertex_Buffer_t>& buf, sycl::queue& q,
                                  float p_R = 0.0f, uint32_t seed = 0)
        : seeds(N_threads), p_R(p_R) {
      std::mt19937 gen(seed);
      // generate random uint32_t numbers
      std::vector<uint32_t> seed_vec(N_threads);
      std::generate(seed_vec.begin(), seed_vec.end(), gen);
      auto tmp_buf = sycl::buffer(seed_vec.data(), seed_vec.size());
      q.submit([&](sycl::handler& h) {
        auto tmp_acc = tmp_buf.get_access<sycl::access::mode::read>(h);
        auto seeds_acc = seeds.get_access<sycl::access::mode::write>(h);
        h.parallel_for(sycl::range<1>(N_threads),
                       [=](sycl::id<1> id) { seeds_acc[id] = tmp_acc[id]; });
      });
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
}  // namespace Sycl_Graph::Epidemiological

#endif