#ifndef SYCL_GRAPH_EPIDEMIOLOGICAL_SIR_INDIVIDUAL_OPERATIONS_HPP
#define SYCL_GRAPH_EPIDEMIOLOGICAL_SIR_INDIVIDUAL_OPERATIONS_HPP
#include <Static_RNG/distributions.hpp>
#include <Sycl_Graph/Algorithms/Operations/Operations.hpp>
#include <Sycl_Graph/Epidemiological/SIR_Individual/Types.hpp>
#include <Sycl_Graph/Graph/Base/Graph_Types.hpp>
#include <random>
namespace Sycl_Graph::Epidemiological
{
using namespace Sycl_Graph::Sycl;
template <Sycl::Vertex_Buffer_type Vertex_Buffer_t>
struct SIR_Individual_Recovery_Op
    : public Vertex_Extract_Operation<Vertex_Buffer_t, SIR_Individual_Recovery_Op<Vertex_Buffer_t>>
{
    typedef SIR_Individual_State_t Target_t;
    typedef typename Vertex_Buffer_t::Vertex_t Vertex_t;
    static constexpr std::array<sycl::access::mode, 1> custom_access_modes = {sycl::access_mode::atomic};
    float p_R = 0.0f;
    SIR_Individual_Recovery_Op(const Vertex_Buffer_t &buf, float p_R = 0.0f) : p_R(p_R)
    {
    }

    void invoke(const auto &v_acc, tuple_like auto &custom_acc, auto &result_acc, sycl::handler &h)
    {
        auto seed_acc = std::get<0>(custom_acc);
        const float p_R = this->p_R;
        auto N_vertices = v_acc.size();
        auto N_threads = std::min({N_vertices, seed_acc.size()});
        auto N_per_thread = N_vertices > N_threads ? N_vertices / N_threads : 1;
        h.parallel_for(seed_acc.size(), [=](sycl::id<1> id) {
            auto seed = sycl::atomic_fetch_add<uint32_t>(seed_acc[id], 1);
            Static_RNG::default_rng rng(seed);
            for (int i = 0; i < N_per_thread; i++)
            {
                auto idx = id*N_per_thread + i;
                if (idx > N_vertices)
                    return;
                Static_RNG::bernoulli_distribution<float> dist(p_R);
                result_acc[idx] = v_acc.data[idx];
                result_acc[idx] = dist(rng) ? SIR_INDIVIDUAL_R : v_acc.data[idx];
            }
        });
    }
    template <typename Graph_t>
    size_t target_buffer_size(const Graph_t &G) const
    {
        auto size = G.template current_size<typename Vertex_Buffer_t::Vertex_t>();
        return size;
    }
};

// Individual Infection Op: Chained with Individual Recovery Op as an inplace operation
template <Sycl_Graph::Sycl::Edge_Buffer_type Edge_Buffer_t, Sycl_Graph::Sycl::Vertex_Buffer_type Vertex_Buffer_t>
struct SIR_Individual_Infection_Op
    : public Edge_Transform_Operation<Edge_Buffer_t, SIR_Individual_Infection_Op<Edge_Buffer_t, Vertex_Buffer_t>>
{
    using Base_t = Edge_Transform_Operation<Edge_Buffer_t, SIR_Individual_Infection_Op<Edge_Buffer_t, Vertex_Buffer_t>>;
    static constexpr sycl::access::mode target_access_mode = sycl::access::mode::write;
    static constexpr std::array<sycl::access::mode, 1> custom_access_modes = {sycl::access_mode::atomic};
    typedef SIR_Individual_State_t Target_t;
    typedef SIR_Individual_State_t Source_t;
    typedef typename Vertex_Buffer_t::Vertex_t Vertex_t;
    float p_I = 0.0f;
    SIR_Individual_Infection_Op(const Edge_Buffer_t &edge_buf, Vertex_Buffer_t &, float p_I = 0.0f) : p_I(p_I)
    {
    }

    void invoke(const auto &edge_acc,
                const auto &from_acc,
                const auto &to_acc,
                tuple_like auto &custom_acc,
                const auto &source_acc,
                auto &target_acc,
                sycl::handler &h)
    {
        auto seed_acc = std::get<0>(custom_acc);
        uint32_t N_threads = edge_acc.size();
        uint32_t N_edges = edge_acc.size();

        //divide the work among the threads
        auto N_per_thread = N_edges / N_threads + 1;

        h.parallel_for(N_threads, [=, this](sycl::id<1> id) {
            auto seed = sycl::atomic_fetch_add<uint32_t>(seed_acc[id], 1);
            Static_RNG::default_rng rng(seed);
            size_t idx = 0;
            for (size_t i = 0; i < N_per_thread; i++)
            {
                idx = id * N_per_thread + i;
                if (idx >= N_edges)
                    break;
                auto id_from = edge_acc[idx].id.from;
                auto id_to = edge_acc[idx].id.to;
                if (edge_acc[idx].is_valid() && (from_acc.data[id_from] == SIR_INDIVIDUAL_I) &&
                    (to_acc.data[id_to] == SIR_INDIVIDUAL_S))
                {
                    auto p_I = edge_acc[id].data;
                    Static_RNG::bernoulli_distribution<float> dist(p_I);
                    if (dist(rng))
                    {
                        target_acc[id_to] = SIR_INDIVIDUAL_I;
                    }
                }
            }
        });
    }
    template <typename Graph_t>
    size_t target_buffer_size(const Graph_t &G) const
    {
        auto size = G.template current_size<Vertex_t>();
        return size;
    }

    template <typename Graph_t>
    size_t source_buffer_size(const Graph_t &G) const
    {
        auto size = G.template current_size<Vertex_t>();
        return size;
    }
};

// Individual Infection Op: Chained with Individual Recovery Op as an inplace operation
struct SIR_Individual_Population_Count_Extract_Op
    : public Vertex_Extract_Operation<SIR_Individual_Vertex_Buffer_t, SIR_Individual_Population_Count_Extract_Op>
{
    using Base_t = Vertex_Extract_Operation<SIR_Individual_Vertex_Buffer_t, SIR_Individual_Population_Count_Extract_Op>;
    using Base_t::Accessor_Types;
    static constexpr sycl::access::mode target_access_mode = sycl::access::mode::write;
    typedef uint32_t Target_t;

    void invoke(const auto &v_acc, auto &target_acc, sycl::handler &h)
    {
#ifdef EPIDEMIOLOGICAL_POPULATION_COUNT_DEBUG
        sycl::stream out(1024, 256, h);
#endif
        h.single_task([=]() {
            target_acc[0] = 0;
            target_acc[1] = 0;
            target_acc[2] = 0;
            uint32_t N_susceptible = 0;
            uint32_t N_infected = 0;
            uint32_t N_recovered = 0;
            for (int i = 0; i < v_acc.size(); i++)
            {
                if (v_acc.data[i] == SIR_INDIVIDUAL_S)
                {
                    N_susceptible++;
                }
                else if (v_acc.data[i] == SIR_INDIVIDUAL_I)
                {
                    N_infected++;
                }
                else if (v_acc.data[i] == SIR_INDIVIDUAL_R)
                {
                    N_recovered++;
                }
                else
                {
                    #ifdef EPIDEMIOLOGICAL_POPULATION_COUNT_DEBUG
                    out << "Invalid state: " << (uint32_t) v_acc.data[i] << sycl::endl;
                    #endif
                }
            }
            target_acc[0] = N_susceptible;
            target_acc[1] = N_infected;
            target_acc[2] = N_recovered;
#ifdef EPIDEMIOLOGICAL_POPULATION_COUNT_DEBUG
            out << "S: " << target_acc[0] << " I: " << target_acc[1] << " R: " << target_acc[2] << sycl::endl;
#endif
        });
    }
    template <typename Graph_t>
    size_t target_buffer_size(const Graph_t &G) const
    {
        return 3;
    }
};


struct SIR_Individual_Population_Count_Transform_Op
    : public Transform_Operation<SIR_Individual_Population_Count_Transform_Op>
{
    using Base_t = Transform_Operation<SIR_Individual_Population_Count_Transform_Op>;
    static constexpr sycl::access::mode target_access_mode = sycl::access::mode::write;
    typedef uint32_t Target_t;
    typedef SIR_Individual_State_t Source_t;

    void invoke(const auto &source_acc, auto &target_acc, sycl::handler &h)
    {
#ifdef EPIDEMIOLOGICAL_POPULATION_COUNT_DEBUG
        //create sycl::stream
        sycl::stream out(1024, 256, h);
#endif
        h.single_task([=]() {
            target_acc[0] = 0;
            target_acc[1] = 0;
            target_acc[2] = 0;
            for (auto i = 0; i < source_acc.size(); i++)
            {
                if (source_acc[i] == SIR_INDIVIDUAL_S)
                {
                    target_acc[0]++;
                }
                else if (source_acc[i] == SIR_INDIVIDUAL_I)
                {
                    target_acc[1]++;
                }
                else if (source_acc[i] == SIR_INDIVIDUAL_R)
                {
                    target_acc[2]++;
                }
            }
#ifdef EPIDEMIOLOGICAL_POPULATION_COUNT_DEBUG
            out << "S: " << target_acc[0] << " I: " << target_acc[1] << " R: " << target_acc[2] << sycl::endl;
#endif
        });
    }
    template <typename Graph_t>
    size_t target_buffer_size(const Graph_t &G) const
    {
        return 3;
    }
};

} // namespace Sycl_Graph::Epidemiological

#endif
