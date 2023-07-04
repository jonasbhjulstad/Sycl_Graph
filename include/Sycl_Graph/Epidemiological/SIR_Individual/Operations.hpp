#ifndef SYCL_GRAPH_EPIDEMIOLOGICAL_SIR_INDIVIDUAL_OPERATIONS_HPP
#define SYCL_GRAPH_EPIDEMIOLOGICAL_SIR_INDIVIDUAL_OPERATIONS_HPP
#include <Static_RNG/distributions.hpp>
#include <Sycl_Graph/Algorithms/Operations/Operations.hpp>
#include <Sycl_Graph/Epidemiological/SIR_Individual/Types.hpp>
#include <Sycl_Graph/Graph/Base/Graph_Types.hpp>
#include <random>
namespace Sycl_Graph::Epidemiological {
  using namespace Sycl_Graph::Sycl;

  template <typename Source_t = SIR_Individual_Vertex_t, typename Target_t = SIR_Individual_State_t>
  struct SIR_Individual_Recovery
      : public Operation_Base<SIR_Individual_Recovery<Source_t, Target_t>,
                              Read_Accessors_t<Source_t>, Write_Accessors_t<Target_t>,
                              Atomic_Accessors_t<uint32_t>> {
    float p_R = 0.0f;
    size_t N_wg = 256;
    SIR_Individual_Recovery(float p_R, size_t N_wg) : p_R(p_R), N_wg(N_wg) {}

    void invoke(const auto &source_acc, auto &target_acc, auto &seed_acc, sycl::handler &h) {
      const float p_R = this->p_R;
      const auto state_acc = get_vertex_data_accessor<Source_t>(source_acc);
      auto N_vertices = state_acc.size();
      auto N_threads = std::min({N_vertices, seed_acc.size()});
      auto N_per_thread = N_vertices > N_threads ? N_vertices / N_threads : 1;
      h.parallel_for(seed_acc.size(), [=](sycl::id<1> id) {
        auto seed = sycl::atomic_fetch_add<uint32_t>(seed_acc[id], 1);
        Static_RNG::default_rng rng(seed);
        for (int i = 0; i < N_per_thread; i++) {
          auto idx = id * N_per_thread + i;
          if (idx > N_vertices) return;
          Static_RNG::bernoulli_distribution<float> dist(p_R);
          target_acc[idx] = ((state_acc[idx] == SIR_INDIVIDUAL_I) && dist(rng)) ? SIR_INDIVIDUAL_R : state_acc[idx];
        }
      });
    }
    template <typename T, typename Graph_t> int get_buffer_size(const Graph_t &G) const {

      std::string T_name = typeid(T).name();
      if constexpr (std::is_same_v<T, SIR_Individual_State_t>) {
        return G.template current_size<Source_t>();
      } else
        return 0;
    }
  };

  template <typename Source_t = SIR_Individual_Vertex_t, typename Target_t = SIR_Individual_State_t,
            Edge_type Edge_t = SIR_Individual_Edge_t>
  struct SIR_Individual_Infection
      : public Operation_Base<SIR_Individual_Infection<Source_t, Target_t, Edge_t>,
                              Read_Accessors_t<Edge_t, Source_t>, Write_Accessors_t<Target_t>,
                              ReadWrite_Accessors_t<uint32_t>> {
    float p_I = 0.0f;
    size_t N_wg = 256;
    size_t N_pop;
    SIR_Individual_Infection(float p_I, size_t N_wg, size_t N_pop = 0)
        : p_I(p_I), N_wg(N_wg), N_pop(N_pop) {}


    //simple copy operation of states
    void initialize(sycl::handler &h, const auto &edge_acc, const auto &source_acc, auto &target_acc, auto &seed_acc) {
      size_t N_pop = source_acc.size();
      size_t N_threads = std::min({N_pop, N_wg});
      // divide the work among the threads
      auto N_per_thread = N_pop / N_threads + 1;
      const auto state_acc = get_vertex_data_accessor<Source_t>(source_acc);
    std::cout << "Initializing infection target buffer ..." << std::endl;
      h.parallel_for(N_threads, [=, this](sycl::id<1> id) {
        size_t idx = 0;
        for (size_t i = 0; i < N_per_thread; i++) {
          idx = id * N_per_thread + i;
          if (idx >= N_pop) break;
          target_acc[idx] = state_acc[idx];
        }
      });
    }


    void invoke(const auto &edge_acc, const auto &source_acc, auto &target_acc, auto &seed_acc,
                sycl::handler &h) {
      size_t N_edges = edge_acc.size();
      size_t N_threads = std::min({N_edges, N_wg});

      // divide the work among the threads
      auto N_per_thread = N_edges / N_threads + 1;
      const auto state_acc = get_vertex_data_accessor<Source_t>(source_acc);

      h.parallel_for(N_threads, [=, this](sycl::id<1> id) {
        auto seed = seed_acc[id]++;
        Static_RNG::default_rng rng(seed);
        size_t idx = 0;
        for (size_t i = 0; i < N_per_thread; i++) {
          idx = id * N_per_thread + i;
          if (idx >= N_edges) break;
          auto id_from = edge_acc[idx].id.from;
          auto id_to = edge_acc[idx].id.to;
          if (edge_acc[idx].is_valid() && (state_acc[id_from] == SIR_INDIVIDUAL_I)
              && (state_acc[id_to] == SIR_INDIVIDUAL_S)) {
            auto p_I = edge_acc[id].data;
            Static_RNG::bernoulli_distribution<float> dist(p_I);
            if (dist(rng)) {
              target_acc[id_to] = SIR_INDIVIDUAL_I;
            }
          }
        }
      });
    }


    template <typename T, typename Graph_t> int get_buffer_size(const Graph_t &G) const {
      if constexpr (std::is_same_v<T, uint32_t>) return N_wg;
      if (is_Vertex_type<T>)
        return G.template current_size<Source_t>();
      else
        return N_pop;
    }
  };  // namespace Sycl_Graph::Epidemiological

  // Individual Infection Op: Chained with Individual Recovery Op as an inplace operation
  template <typename Source_t = SIR_Individual_Vertex_t> struct SIR_Individual_Population_Count
      : public Operation_Base<SIR_Individual_Population_Count<Source_t>, Read_Accessors_t<Source_t>, ReadWrite_Accessors_t<uint32_t>, std::tuple<>> {
    size_t N_pop = 0;
    SIR_Individual_Population_Count(size_t N_pop = 0) : N_pop(N_pop) {}
    void invoke(const auto &source_acc, auto &target_acc, sycl::handler &h) {
#ifdef EPIDEMIOLOGICAL_POPULATION_COUNT_DEBUG
      sycl::stream out(1024, 256, h);
#endif
      h.single_task([=]() {
        uint32_t N_susceptible = 0;
        uint32_t N_infected = 0;
        uint32_t N_recovered = 0;
        const auto &state_acc = get_vertex_data_accessor<Source_t>(source_acc);
        for (int i = 0; i < state_acc.size(); i++) {
          if (state_acc[i] == SIR_INDIVIDUAL_S) {
            N_susceptible++;
          } else if (state_acc[i] == SIR_INDIVIDUAL_I) {
            N_infected++;
          } else if (state_acc[i] == SIR_INDIVIDUAL_R) {
            N_recovered++;
          } else {
#ifdef EPIDEMIOLOGICAL_POPULATION_COUNT_DEBUG
            out << "Invalid state: " << (uint32_t)state_acc[i] << "at idx: " << i << sycl::endl;
#endif
          }
        }
        target_acc[0] = N_susceptible;
        target_acc[1] = N_infected;
        target_acc[2] = N_recovered;

#ifdef EPIDEMIOLOGICAL_POPULATION_COUNT_DEBUG
        out << "S: " << target_acc[0] << " I: " << target_acc[1] << " R: " << target_acc[2]
            << sycl::endl;
#endif
      });
    }
    template <typename T, typename Graph_t> int get_buffer_size(const Graph_t &G) const {
      if (N_pop == 0 && is_Vertex_type<T>)
        return G.template current_size<Source_t>();
      else
        return 3;
    }
  };
}

#endif
