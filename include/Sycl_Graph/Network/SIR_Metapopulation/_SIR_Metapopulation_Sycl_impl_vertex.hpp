#ifndef SIR_METAPOPULATION_NETWORK_SYCL_IMPL_HPP
#define SIR_METAPOPULATION_NETWORK_SYCL_IMPL_HPP
#include "Sycl_Graph/path_config.hpp"
#include <Sycl_Graph/Tracy_Config.hpp>
#define SYCL_FLOAT_PRECISION 32
#include "SIR_Metapopulation_Types.hpp"
#include <Sycl_Graph/Graph/Sycl/Invariant_Graph.hpp>
#include <Sycl_Graph/Network/Network.hpp>
#include <Sycl_Graph/Math/math.hpp>
#include <Static_RNG/distributions.hpp>
#include <Sycl_Graph/Network/SIR_Metapopulation/SIR_Metapopulation_Types.hpp>
#include <CL/sycl.hpp>
#include <fmt/format.h>
#include <random>
#include <stddef.h>
#include <type_traits>
#include <utility>
#include <filesystem>

template <>
struct sycl::is_device_copyable<
    Sycl_Graph::Network_Models::SIR_Metapopulation_Node_Param>
    : std::true_type
{
};

template <>
struct sycl::is_device_copyable<
    Sycl_Graph::Network_Models::SIR_Metapopulation_Param> : std::true_type
{
};

template <>
struct sycl::is_device_copyable<
    Sycl_Graph::Network_Models::SIR_Metapopulation_State> : std::true_type
{
};
namespace Sycl_Graph
{

  namespace Sycl::Network_Models
  {

    float compute_infection_probability(float beta, float N_I, float N, float dt,
                                        float c = 1.f)
    {
      return 1 - std::exp(-beta * N_I * dt / N);
    }

    // sycl::is_device_copyable_v<SIR_Metapopulation_State>
    // is_copyable_SIR_Invidual_State;
    using namespace Sycl_Graph::Network_Models;
    using namespace Static_RNG;

    using SIR_Metapopulation_Graph =
        Sycl_Graph::Sycl::Graph<SIR_Metapopulation_Node, SIR_Metapopulation_Param,
                                uint32_t>;
    template <Static_RNG::rng_type RNG = Static_RNG::default_rng>
    struct SIR_Metapopulation_Network
        : public Network<SIR_Metapopulation_Network<RNG>, SIR_Metapopulation_State,
                         SIR_Metapopulation_Temporal_Param>
    {
      // convert to typedef
      typedef SIR_Metapopulation_Graph Graph_t;
      typedef typename Graph_t::Vertex_t Vertex_t;
      typedef typename Graph_t::Edge_t Edge_t;
      typedef Network<SIR_Metapopulation_Network<RNG>,
                      SIR_Metapopulation_Temporal_Param, SIR_Metapopulation_State>
          Base_t;

      sycl::queue &q;
      Graph_t &G;
      static constexpr auto invalid_id = Graph_t::invalid_id;

      const uint32_t t = 0;
      // Poisson approximation ratio (np <= poisson_approx_ratio)
      float poisson_approx_ratio = 10.f;
      sycl::buffer<RNG, 1> rng_buf;
      const std::vector<uint32_t> N_pop;
      const std::vector<normal_distribution<float>> I0_dist;
      const std::vector<normal_distribution<float>> R0_dist;
      const std::vector<float> alpha_0;
      const std::vector<float> node_beta_0;
      const std::vector<float> edge_beta_0;

      SIR_Metapopulation_Network(Graph_t &G, const std::vector<uint32_t> &N_pop,
                                 const std::vector<normal_distribution<float>> &I0,
                                 const std::vector<float> &alpha,
                                 const std::vector<float> &node_beta,
                                 const std::vector<float> &edge_beta,
                                 int seed = 777)
          : SIR_Metapopulation_Network(
                G, N_pop, I0, std::vector<normal_distribution<float>>(I0.size()),
                alpha, node_beta, edge_beta, seed) {}

      SIR_Metapopulation_Network(Graph_t &G, const std::vector<uint32_t> &N_pop,
                                 const std::vector<normal_distribution<float>> I0,
                                 const std::vector<normal_distribution<float>> R0,
                                 const std::vector<float> &alpha,
                                 const std::vector<float> &node_beta,
                                 const std::vector<float> &edge_beta,
                                 int seed = 777)
          : q(G.q), G(G), N_pop(N_pop), I0_dist(I0), R0_dist(R0),
            rng_buf(sycl::range<1>(std::max({G.N_vertices(), G.N_edges()}))), alpha_0(alpha),
            node_beta_0(node_beta), edge_beta_0(edge_beta)
      {

        generate_seeds(seed);
        construction_debug_report();
      }

      void initialize()
      {
        ZoneScoped;
        const uint32_t N_vertices = G.N_vertices();
        sycl::buffer<uint32_t, 1> N_pop_buf(N_pop);

        sycl::buffer<normal_distribution<float>, 1> I0_dist_buf(I0_dist);
        sycl::buffer<normal_distribution<float>, 1> R0_dist_buf(R0_dist);
        q.submit([&](sycl::handler &h)
                 {
      auto N_pop_acc = N_pop_buf.get_access<sycl::access::mode::read>(h);
      auto rng_acc =
          rng_buf.template get_access<sycl::access::mode::read_write>(h);
      auto v = G.vertex_buf.template get_access<sycl::access::mode::write>(h);
      auto I0_dist_acc =
          I0_dist_buf.template get_access<sycl::access::mode::read_write>(h);
      auto R0_dist_acc =
          R0_dist_buf.template get_access<sycl::access::mode::read_write>(h);
      sycl::stream out(1024, 256, h);
      h.parallel_for(sycl::range<1>(N_vertices), [=](sycl::id<1> id) {
        // total population stored in susceptible state
        auto N_pop = N_pop_acc[id];
        auto &rng = rng_acc[id];
        auto& I0_dist = I0_dist_acc[id];
        auto& R0_dist = R0_dist_acc[id];

        uint32_t I0 = I0_dist(rng);
        uint32_t R0 = R0_dist(rng);

        auto &state = v.data[id].state;
        state.S = N_pop - I0 - R0;
        state.I = I0;
        state.R = R0;

      }); });

        assert_population_size("Initialization");
        set_edge_beta(edge_beta_0);
        set_node_beta(node_beta_0);
        set_alpha(alpha_0);
      }

      void set_alpha(const std::vector<float> &alpha)
      {
        set_alpha(alpha, Sycl_Graph::range(0, alpha.size()));
      }

      void set_alpha(const std::vector<float> &alpha,
                     const std::vector<uint32_t> &idx)
      {
        ZoneScoped;
        sycl::buffer<float, 1> alpha_buf(alpha);
        sycl::buffer<uint32_t, 1> idx_buf(idx);
        q.submit([&](sycl::handler &h)
                 {
      auto alpha_acc = alpha_buf.get_access<sycl::access::mode::read>(h);
      auto idx_acc = idx_buf.get_access<sycl::access::mode::read>(h);
      auto v = G.vertex_buf.template get_access<sycl::access::mode::write>(h);
      h.parallel_for(sycl::range<1>(idx_acc.size()), [=](sycl::id<1> id) {
        v.data[idx_acc[id]].param.alpha = alpha_acc[id];
      }); });
      }

      void set_node_beta(const std::vector<float> &beta)
      {
        set_node_beta(beta, Sycl_Graph::range(0, beta.size()));
      }

      void set_node_beta(const std::vector<float> &beta,
                         const std::vector<uint32_t> &idx)
      {
        ZoneScoped;

        sycl::buffer<float, 1> beta_buf(beta);
        sycl::buffer<uint32_t, 1> idx_buf(idx);
        q.submit([&](sycl::handler &h)
                 {
      auto beta_acc = beta_buf.get_access<sycl::access::mode::read>(h);
      auto idx_acc = idx_buf.get_access<sycl::access::mode::read>(h);
      auto v = G.vertex_buf.template get_access<sycl::access::mode::write>(h);
      h.parallel_for(sycl::range<1>(idx.size()), [=](sycl::id<1> id) {
        v.data[idx_acc[id]].param.beta = beta_acc[id];
      }); });
      }

      void set_edge_beta(const std::vector<float> &beta)
      {
        ZoneScoped;

        if (G.N_edges() == 0)
        {
          std::cout << "Warning: Unable to set edge beta, graph has no edges."
                    << std::endl;
          return;
        }
        sycl::buffer<float, 1> beta_buf(beta);
        q.submit([&](sycl::handler &h)
                 {
      auto beta_acc = beta_buf.get_access<sycl::access::mode::read>(h);
      auto e_acc = G.edge_buf.template get_access<sycl::access::mode::write>(h);
      h.parallel_for(sycl::range<1>(beta_acc.size()), [=](sycl::id<1> id) {
        e_acc.data[id].beta = beta_acc[id];
      }); });
      }

      void set_edge_beta(const std::vector<float> &beta,
                         const std::vector<uint32_t> &to_idx,
                         const std::vector<uint32_t> &from_idx)
      {
        ZoneScoped;

        if (G.N_edges() == 0)
        {
          std::cout << "Warning: Unable to set edge beta, graph has no edges."
                    << std::endl;
          return;
        }
        sycl::buffer<float, 1> beta_buf(beta);
        sycl::buffer<uint32_t, 1> to_idx_buf(to_idx);
        sycl::buffer<uint32_t, 1> from_idx_buf(from_idx);
        uint32_t N_vertices = G.N_vertices();
        q.submit([&](sycl::handler &h)
                 {
      auto beta_acc = beta_buf.get_access<sycl::access::mode::read>(h);
      auto to_idx_acc = to_idx_buf.get_access<sycl::access::mode::read>(h);
      auto from_idx_acc = from_idx_buf.get_access<sycl::access::mode::read>(h);
      auto e_acc = G.edge_buf.get_access<sycl::access::mode::write>(h);
      h.parallel_for(sycl::range<1>(to_idx_acc.size()), [=](sycl::id<1> id) {
        for (int i = 0; i < N_vertices; i++) {
          if (e_acc.to[i] == to_idx_acc[id] &&
              e_acc.from[i] == from_idx_acc[id])
            e_acc.data[i].beta = beta_acc[id];
        }
      }); });
      }

      SIR_Metapopulation_State read_state(SIR_Metapopulation_Temporal_Param tp)
      {
        ZoneScoped;

        SIR_Metapopulation_State state;
        sycl::buffer<SIR_Metapopulation_State, 1> state_buf(&state,
                                                            sycl::range<1>(1));
        // set state_buf to 0
        q.submit([&](sycl::handler &h)
                 {
      auto state_acc = state_buf.get_access<sycl::access::mode::write>(h);
      h.single_task([=] { state_acc[0] = SIR_Metapopulation_State(); }); });
        const uint32_t N_vertices = G.N_vertices();

        q.submit([&](sycl::handler &h)
                 {
      auto v = G.get_vertex_access<sycl::access::mode::read>(h);
      auto state_acc = state_buf.get_access<sycl::access::mode::write>(h);

      h.single_task([=] {
        for (int i = 0; i < N_vertices; i++) {
          state_acc[0] += v.data[i].state;
        }
      }); });
        q.wait();

        return state;
      }

      std::vector<SIR_Metapopulation_State> read_node_states(const SIR_Metapopulation_Temporal_Param tp)
      {
        ZoneScoped;
        const auto N_vertices = G.N_vertices();
        std::vector<SIR_Metapopulation_State> states(N_vertices);
        sycl::buffer<SIR_Metapopulation_State, 1> state_buf(states.data(), sycl::range<1>(N_vertices));
        auto event = q.submit([&](sycl::handler &h)
                              {
      auto v = G.get_vertex_access<sycl::access::mode::read>(h);
      auto state_acc = state_buf.get_access<sycl::access::mode::write>(h);
        h.parallel_for(sycl::range<1>(N_vertices), [=](sycl::id<1> id) {
          state_acc[id] = v.data[id].state;
        }); });
        event.wait();
        return states;
      }
      void advance(SIR_Metapopulation_Temporal_Param tp)
      {
        ZoneScoped;

        auto inf_bufs = infection_scatter(tp.dt);
        auto rec_buf = recovery_scatter(tp.dt);

        gather(inf_bufs.first, inf_bufs.second, rec_buf);
      }

      bool terminate(SIR_Metapopulation_State x,
                     const SIR_Metapopulation_Temporal_Param tp)
      {
        return false;
      }

    private:
      // function for infection step
      std::pair<sycl::buffer<uint32_t, 1>, sycl::buffer<uint32_t, 1>>
      infection_scatter(float dt)
      {
        ZoneScoped;
        const uint32_t N_vertices = G.N_vertices();
        const uint32_t N_edges = G.N_edges();
        using Sycl_Graph::Network_Models::SIR_Metapopulation_State;

        sycl::buffer<uint32_t, 1> v_inf_buf((sycl::range<1>(N_vertices)));
        FrameMarkStart("Vertex Infections");
        auto event_v_inf = q.submit([&](sycl::handler &h)
                                    {
      auto v_acc = G.get_vertex_access<sycl::access::mode::read>(h);
      auto v_inf_acc =
          v_inf_buf.template get_access<sycl::access::mode::read_write>(h);
      auto rng_acc =
          rng_buf.template get_access<sycl::access::mode::read_write>(h);

      const float poisson_approx_ratio = this->poisson_approx_ratio;
      h.parallel_for(sycl::range<1>(N_vertices), [=](sycl::id<1> id) {
        auto beta = v_acc.data[id].param.beta;
        auto S = v_acc.data[id].state.S;
        auto I = v_acc.data[id].state.I;
        auto R = v_acc.data[id].state.R;
        auto N_pop = S + I + R;
        auto p_I = compute_infection_probability(beta, I, N_pop, dt);
        Static_RNG::approximate_binomial_distribution dist(I, p_I, poisson_approx_ratio);
        v_inf_acc[id] = dist(rng_acc[id]);
      }); });
        event_v_inf.wait();
        assert_population_size("Vertex Infections");
        FrameMarkEnd("Vertex Infections");
        sycl::buffer<uint32_t, 1> e_inf_buf(0, sycl::range<1>(N_edges));

        if (N_edges > 0)
        {

          const float poisson_approx_ratio = this->poisson_approx_ratio;

          FrameMarkStart("Edge Infections");
          auto event_e_inf = q.submit([&](sycl::handler &h)
                                      {
        auto v_acc = G.get_vertex_access<sycl::access::mode::read>(h);
        auto e_acc = G.get_edge_access<sycl::access::mode::read>(h);
        auto rng_acc =
            rng_buf.template get_access<sycl::access::mode::read_write>(h);
        auto e_inf_acc =
            e_inf_buf.template get_access<sycl::access::mode::write>(h);
        sycl::stream out(1024, 256, h);
        h.parallel_for(sycl::range<1>(N_edges), [=](sycl::id<1> edge_idx) {
          
          const auto from_id = e_acc.from[edge_idx];
          const auto to_id = e_acc.to[edge_idx];
          if(from_id == invalid_id || to_id == invalid_id) return;

          const auto& beta = e_acc.data[edge_idx].beta;
          const auto& S = v_acc.data[to_id].state.S;
          const auto& I = v_acc.data[from_id].state.I;
          
          auto p_I = compute_infection_probability(beta, I, S+I, dt);
        Static_RNG::approximate_binomial_distribution dist(S, p_I, poisson_approx_ratio);
          uint32_t N_potential_inf = dist(rng_acc[edge_idx]);

          auto to_state = v_acc.data[e_acc.to[edge_idx]].state;
          auto from_state = v_acc.data[e_acc.from[edge_idx]].state;
          float susceptible_frac = (float)S / (float)(to_state.S + to_state.I + to_state.R);
          Static_RNG::approximate_binomial_distribution<> d_edge(N_potential_inf, susceptible_frac, poisson_approx_ratio);
          uint32_t N_edge_infected = d_edge(rng_acc[edge_idx]);
          if (to_id == 0 || to_id == 1)
            out << "Edge " << edge_idx << " from " << from_id << " to " << to_id << " has " << N_edge_infected << " infections" << sycl::endl;
          e_inf_acc[edge_idx] = N_edge_infected;

        }); });

          event_e_inf.wait();
          assert_population_size("Edge Infections");
          FrameMarkEnd("Edge Infections");
        }

        return std::make_pair(v_inf_buf, e_inf_buf);
      }
      sycl::buffer<uint32_t, 1> recovery_scatter(float dt)
      {
        ZoneScoped;
        const uint32_t N_vertices = G.N_vertices();
        sycl::buffer<uint32_t, 1> v_rec_buf((sycl::range<1>(N_vertices)));
        const float poisson_approx_ratio = this->poisson_approx_ratio;
        auto event_v_rec = q.submit([&](sycl::handler &h)
                                    {
      auto v_acc = G.get_vertex_access<sycl::access::mode::read>(h);
      auto v_rec_acc =
          v_rec_buf.template get_access<sycl::access::mode::read_write>(h);
      auto rng_acc =
          rng_buf.template get_access<sycl::access::mode::read_write>(h);
      h.parallel_for(sycl::range<1>(N_vertices), [=](sycl::id<1> id) {
        auto alpha = v_acc.data[id].param.alpha;
        auto p_R = 1 - sycl::exp(-alpha * dt);
        auto I = v_acc.data[id].state.I;
        Static_RNG::approximate_binomial_distribution dist(I, p_R, poisson_approx_ratio);
        v_rec_acc[id] = dist(rng_acc[id]);
      }); });
        event_v_rec.wait();

        return v_rec_buf;
      }
      void gather(sycl::buffer<uint32_t, 1> &v_inf_buf,
                  sycl::buffer<uint32_t, 1> &e_inf_buf,
                  sycl::buffer<uint32_t, 1> &v_rec_buf)
      {
        ZoneScoped;
        const uint32_t N_vertices = G.N_vertices();
        const uint32_t N_edges = G.N_edges();
        const float poisson_approx_ratio = this->poisson_approx_ratio;
        auto event_v_gather = q.submit([&](sycl::handler &h)
                                       {
  
      auto v_acc = G.get_vertex_access<sycl::access::mode::read_write>(h);
      auto v_inf_acc =
          v_inf_buf.template get_access<sycl::access::mode::read>(h);
      auto v_rec_acc =
          v_rec_buf.template get_access<sycl::access::mode::read>(h);
      h.parallel_for(sycl::range<1>(N_vertices), [=](sycl::id<1> id) {
        // Vertex infections and recoveries are updated first

        auto state = v_acc.data[id].state;
        uint32_t delta_I = std::min<uint32_t>(state.S, v_inf_acc[id]);
        uint32_t delta_R = std::min<uint32_t>(state.I, v_rec_acc[id]);
        state.S -= delta_I;
        state.I += delta_I - delta_R;
        state.R += delta_R;
      }); });
        event_v_gather.wait();
        assert_population_size("Vertex Gather");

        auto event_e_gather = q.submit([&](sycl::handler &h)
                                       {
                                        sycl::stream out(1024, 256, h);
                                        auto v_acc = G.get_vertex_access<sycl::access::mode::read_write>(h);
                                        auto e_acc = G.get_edge_access<sycl::access::mode::read>(h);
                                        auto e_inf_acc = e_inf_buf.template get_access<sycl::access::mode::read>(h);
                                        h.parallel_for(sycl::range<1>(N_vertices), [=](sycl::id<1> id) {
                                          uint32_t I_edge_total = 0;
                                          for(int i = 0; i < N_edges; i++)
                                          {
                                            if(e_acc.to[i] == id)
                                            {
                                              I_edge_total += e_inf_acc[i];
                                            }
                                          }
                                          // out << "Vertex " << id << " has " << I_edge_total << " infections" << sycl::endl;
                                          auto& state = v_acc.data[id].state;
                                          uint32_t delta_I = std::min<uint32_t>(state.S, I_edge_total);
                                          state.S -= delta_I;
                                          state.I += delta_I;
                                        }); });

        event_e_gather.wait();
        assert_population_size("Edge Gather");
      }

      void reset() { initialize(); }

      void generate_seeds(int seed)
      {
        ZoneScoped;
        if (rng_buf.size() > 0)
        {
          // generate seeds
          std::vector<int> seeds(rng_buf.size());
          // random device
          // mt19937 generator
          std::mt19937_64 gen(seed);
          std::generate(seeds.begin(), seeds.end(), gen);
          sycl::buffer<int, 1> seed_buf(seeds);
          q.submit([&](sycl::handler &h)
                   {
      auto seed_acc = seed_buf.template get_access<sycl::access::mode::read>(h);
      auto rng_acc = rng_buf.template get_access<sycl::access::mode::write>(h);
      h.parallel_for(sycl::range<1>(rng_acc.size()),
                     [=](sycl::id<1> id) { rng_acc[id].seed(seed_acc[id]); }); });
        }
      }

      void construction_debug_report()
      {
#ifdef SYCL_GRAPH_DEBUG
        // fmt open file

        static uint32_t Instance_Count = 0;
        uint32_t debug_instance_ID = Instance_Count++;
        const std::string filename =
            SYCL_GRAPH_LOG_DIR +
            std::string("SIR_Metapopulation/Network" +
                        std::to_string(debug_instance_ID) + ".txt");
        // create directory if it doesn't exist
        std::filesystem::create_directories(SYCL_GRAPH_LOG_DIR +
                                            std::string("SIR_Metapopulation/"));
        std::ofstream file(filename);
        if (!file.is_open())
        {
          std::cout << "Error opening file: " << filename << std::endl;
          return;
        }

        file << "SIR_Metapopulation Network Report for instance "
             << debug_instance_ID << std::endl;
        file << "---------------------------------" << std::endl;
        file << "N_vertices: " << G.N_vertices() << std::endl;
        file << "N_edges: " << G.N_edges() << std::endl;
        file << "---------------------------------" << std::endl;
        file << "Buffer sizes" << std::endl;
        file << "---------------------------------" << std::endl;
        file << "Vertex buffer size: " << G.vertex_buf.byte_size() << std::endl;
        file << "Edge buffer size: " << G.edge_buf.byte_size() << std::endl;
        file << "RNG buffer size: " << rng_buf.byte_size() << std::endl;
        file << "---------------------------------" << std::endl;
        size_t total_passive_size =
            G.vertex_buf.byte_size() + G.edge_buf.byte_size() + rng_buf.byte_size();
        size_t max_active_size =
            total_passive_size + 3 * G.N_vertices() * sizeof(float);
        file << "Total Buffer space requirement: " << max_active_size << std::endl;
        file.close();
#endif
      }
      void assert_population_size(const std::string &event_name)
      {

        // create buffer for event_name
#ifdef SYCL_GRAPH_DEBUG
        sycl::buffer<char, 1> event_name_buf(event_name.data(), sycl::range<1>(event_name.size()));
        sycl::buffer<uint32_t, 1> N_pop_buf(N_pop.data(), sycl::range<1>(N_pop.size()));
        q.submit([&](sycl::handler &h)
                 {
        auto event_name_acc = event_name_buf.template get_access<sycl::access::mode::read>(h);
        auto v_acc = G.get_vertex_access<sycl::access::mode::read>(h);
        auto rng_acc =
            rng_buf.template get_access<sycl::access::mode::read_write>(h);
        auto N_pop_acc =
            N_pop_buf.template get_access<sycl::access::mode::read>(h);
        sycl::stream os(1024, 128, h);
        h.single_task([=]() {
          for(int id = 0; id < v_acc.size(); id++)
          {
          auto state = v_acc.data[id].state;
          auto N_pop = N_pop_acc[id];
          if (state.S + state.I + state.R != N_pop)
          {
            os << "Event: ";
            for(int i = 0; i < event_name_acc.size(); i++)
              os << event_name_acc[i];
            os << sycl::endl;
            os << "Vertex " << id << " has incorrect population size" << sycl::endl;
            os << "State: " << state.S << " " << state.I << " " << state.R << sycl::endl;
            os << "N_pop: " << N_pop << sycl::endl;
          }
          }
        }); });
#endif
      }
    };

  } // namespace Sycl::Network_Models
} // namespace Sycl_Graph
#endif