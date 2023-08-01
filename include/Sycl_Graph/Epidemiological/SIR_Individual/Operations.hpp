#ifndef SYCL_GRAPH_EPIDEMIOLOGICAL_SIR_INDIVIDUAL_OPERATIONS_HPP
#define SYCL_GRAPH_EPIDEMIOLOGICAL_SIR_INDIVIDUAL_OPERATIONS_HPP
#include <Static_RNG/distributions.hpp>
#include <Sycl_Graph/Algorithms/Operations/Operations.hpp>
#include <Sycl_Graph/Epidemiological/SIR_Individual/Types.hpp>
#include <Sycl_Graph/Graph/Base/Graph_Types.hpp>
#include <random>
namespace Sycl_Graph::Epidemiological {
  using namespace Sycl_Graph::Sycl;

  struct SIR_Individual_Recovery
      : public Operation_Base<SIR_Individual_Recovery,
                              Accessor_t<SIR_Individual_State_t, sycl::access_mode::read>,
                              Accessor_t<SIR_Individual_State_t, sycl::access_mode::write>,
                              Accessor_t<uint32_t, sycl::access_mode::atomic>> {
    float p_R = 0.0f;
    size_t N_wg = 256;
    SIR_Individual_Recovery(float p_R, size_t N_wg) : p_R(p_R), N_wg(N_wg) {
      this->template set_size<uint32_t>(N_wg);
    }

    void invoke(sycl::handler &h, const auto &source_acc, auto &target_acc, auto &seed_acc) {
      const float p_R = this->p_R;
      auto N_vertices = source_acc.size();
      auto N_threads = std::min({N_vertices, seed_acc.size()});
      auto N_per_thread = N_vertices > N_threads ? N_vertices / N_threads : 1;
      h.parallel_for(seed_acc.size(), [=](sycl::id<1> id) {
        auto seed = sycl::atomic_fetch_add<uint32_t>(seed_acc[id], 1);
        Static_RNG::default_rng rng(seed);
        for (int i = 0; i < N_per_thread; i++) {
          auto idx = id * N_per_thread + i;
          if (idx > N_vertices) return;
          Static_RNG::bernoulli_distribution<float> dist(p_R);
          target_acc[idx] = ((source_acc[idx] == SIR_INDIVIDUAL_I) && dist(rng)) ? SIR_INDIVIDUAL_R
                                                                                : source_acc[idx];
        }
      });
    }
  };

  struct SIR_Individual_Infection
      : public Operation_Base<SIR_Individual_Infection,
                              Accessor_t<std::pair<uint32_t, uint32_t>, sycl::access_mode::read>,
                              Accessor_t<float, sycl::access_mode::read>,
                              Accessor_t<SIR_Individual_State_t, sycl::access_mode::read>,
                              Accessor_t<SIR_Individual_State_t, sycl::access_mode::write>,
                              Accessor_t<uint32_t, sycl::access_mode::read_write>> {
    float p_I = 0.0f;
    size_t N_wg = 256;
    size_t N_pop;
    SIR_Individual_Infection(float p_I, size_t N_wg, size_t N_pop = 0)
        : p_I(p_I), N_wg(N_wg), N_pop(N_pop) {
      this->template set_size<uint32_t>(N_wg);
      this->template set_size<0>(N_pop);
      this->template set_size<1>(N_pop);
    }

    void invoke(sycl::handler &h, const auto &edge_id_acc, const auto& p_I_acc, const auto &source_acc, auto &target_acc,
                auto &seed_acc) {
      size_t N_edges = edge_id_acc.size();
      size_t N_threads = std::min({N_edges, N_wg});

      // divide the work among the threads
      auto N_per_thread = N_edges / N_threads + 1;

      auto is_valid = [](auto edge_id) { return (edge_id.first != std::numeric_limits<uint32_t>::max()) && (edge_id.second != std::numeric_limits<uint32_t>::max()); };

      h.parallel_for(N_threads, [=, this](sycl::id<1> id) {
        auto seed = seed_acc[id]++;
        Static_RNG::default_rng rng(seed);
        size_t idx = 0;
        for (size_t i = 0; i < N_per_thread; i++) {
          idx = id * N_per_thread + i;
          if (idx >= N_edges) break;
          auto id_from = edge_id_acc[idx].first;
          auto id_to = edge_id_acc[idx].second;
          if (is_valid(edge_id_acc[idx])  && (source_acc[id_from] == SIR_INDIVIDUAL_I)
              && (source_acc[id_to] == SIR_INDIVIDUAL_S)) {
            auto p_I = p_I_acc[id];
            Static_RNG::bernoulli_distribution<float> dist(p_I);
            if (dist(rng)) {
              target_acc[id_to] = SIR_INDIVIDUAL_I;
            }
          }
        }
      });
    }

  };  // namespace Sycl_Graph::Epidemiological


  // Individual Infection Op: Chained with Individual Recovery Op as an inplace operation
  struct SIR_Individual_Population_Count
      : public Operation_Base<SIR_Individual_Population_Count,
                              Accessor_t<SIR_Individual_State_t, sycl::access_mode::read>,
                              Accessor_t<SIR_Individual_State_t, sycl::access_mode::write>> {
    size_t N_pop = 0;
    SIR_Individual_Population_Count(size_t N_pop = 0) : N_pop(N_pop) {
    }
    void invoke(sycl::handler &h, const auto &source_acc, auto &target_acc) {
#ifdef EPIDEMIOLOGICAL_POPULATION_COUNT_DEBUG
      sycl::stream out(1024, 256, h);
#endif
      using SIR_State_Acc_t = sycl::accessor<SIR_Individual_State_t, 1, sycl::access_mode::read>;
      h.single_task([=]() {
        uint32_t N_susceptible = 0;
        uint32_t N_infected = 0;
        uint32_t N_recovered = 0;

        for (int i = 0; i < source_acc.size(); i++) {
          if (source_acc[i] == SIR_INDIVIDUAL_S) {
            N_susceptible++;
          } else if (source_acc[i] == SIR_INDIVIDUAL_I) {
            N_infected++;
          } else if (source_acc[i] == SIR_INDIVIDUAL_R) {
            N_recovered++;
          } else {
#ifdef EPIDEMIOLOGICAL_POPULATION_COUNT_DEBUG
            out << "Invalid state: " << (uint32_t)source_acc[i] << "at idx: " << i << sycl::endl;
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
        return 3;
    }
  };

  struct SIR_State_Injection
      : public Operation_Base<
      SIR_State_Injection,
      Accessor_t<SIR_Individual_State_t, sycl::access_mode::read>,
                              Accessor_t<SIR_Individual_Vertex_t, sycl::access_mode::write>> {
    void invoke(sycl::handler &h, const auto &source_acc, auto &target_acc) {
      h.parallel_for(source_acc.size(), [=](auto id) {
        target_acc[id] = source_acc[id];
      });
    }
  };


}  // namespace Sycl_Graph::Epidemiological

#endif
