#ifndef SYCL_GRAPH_NETWORK_DYNAMICS_NETWORK_HPP
#define SYCL_GRAPH_NETWORK_DYNAMICS_NETWORK_HPP
#include <Sycl_Graph/Graph/Base/Graph.hpp>


namespace Sycl_Graph::Network::Dynamic
{
#include <Sycl_Graph/Math/math.hpp>
#include <Sycl_Graph/Graph/Base/Graph.hpp>
#include <random>
#include <stddef.h>
#include <vector>
#include <limits>

namespace Sycl_Graph::Network::Dynamics
{

    template <Sycl_Graph::Graph::Invariant::Graph_type Graph_t, class Derived>
    struct Network: public Graph_t
    {

        typedef typename Graph_t::Vertex_Buffer_t Vertex_Buffer_t;
        typedef Vertex_Buffer_t Vertex_Sync_Buffer_t;
        typedef typename Graph_t::Edge_Buffer_t Edge_Buffer_t;
        typedef typename Vertex_Buffer_t::Data_t State_t;
        typedef typename Edge_Buffer_t::TemporalParam_t TemporalParam_t;
        Vertex_Buffer_t vertex_sync_buf;
        Edge_Buffer_t edge_sync_buf;

        Network() = default;
        Network(const Vertex_Buffer_t& vertex_buffer, const Edge_Buffer_t& edge_buffer) : Graph_t(vertex_buffer, edge_buffer) {}
        
        bool advance()
        {
            return static_cast<Derived *>(this)->advance();
        }


        template <Vertex_type ... Vs>
        auto read_node_states(const TemporalParam_t& tp){
            return static_cast<Derived *>(this)->read_node_states<Vs ...>(tp);
        }

        template <Edge_type ... Es>
        auto read_link_states(const TemporalParam_t& tp){
            return static_cast<Derived *>(this)->template read_edge_states<Es ...>(tp);
        }

        auto read_state(const TemporalParam_t& tp){
            return static_cast<Derived *>(this)->read_state(tp);
        }
        void reset() { static_cast<Derived *>(this)->reset(); }

        bool terminate(const State_t &x, const TemporalParam_t& tp) const
        {
            return static_cast<Derived *>(this)->terminate(x, tp);
        }
        
        std::vector<State_t> simulate(uint32_t Nt = std::numeric_limits<uint32_t>::max(), std::vector<TemporalParam> tp = {})
        {
            return static_cast<Derived *>(this)->simulate(Nt, tp);
            // std::vector<State_t> trajectory(Nt + 1);
            // if (tp.size() < Nt+1)
            //     tp.resize(Nt+1);
            // //reserve space for the trajectories
            // uint32_t t = 0;

            // TemporalParam tp_i = tp[0];
            // trajectory[0] = read_state(tp_i);
            // for (int i = 0; i < Nt; i++)
            // {
            //     tp_i = (tp.size() > 0) ? tp[i] : TemporalParam();
            //     advance(tp_i);
            //     trajectory[i + 1] = read_state(tp[i+1]);
            //     if (terminate(trajectory[i + 1], tp_i))
            //     {
            //         break;
            //     }
            // }
            return trajectory;
        }

    };    

} // namespace Sycl_Graph::Network::Dynamic

#endif