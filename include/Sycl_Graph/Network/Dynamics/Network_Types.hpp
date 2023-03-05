#ifndef SYCL_GRAPH_NETWORK_DYNAMIC_NETWORK_TYPES_HPP
#define SYCL_GRAPH_NETWORK_DYNAMIC_NETWORK_TYPES_HPP
#include <Sycl_Graph/Graph/Base/Graph_types.hpp>
namespace Sycl_Graph::Network::Dynamics
{
    template <typename D, typename _ID_t = uint32_t, _ID_t _invalid_id = std::numeric_limits<_ID_t>::max()>
    using Vertex = Sycl_Graph::Invariant::Vertex<D, _ID_t, _invalid_id>;

    template <Sycl_Graph::Invariant::Vertex_type V, typename _Temporal_Param_t>
    struct Node: public V
    {
        typedef _Temporal_Param_t Temporal_Param_t;
        void advance(const Temporal_Param_t& tp)
        {

        }
    };

    template <Sycl_Graph::Invariant::Edge_type E, typename _Temporal_Param_t, typename Derived>
    struct Link: public E
    {
        typedef _Temporal_Param_t Temporal_Param_t;
        typedef sycl::accessor<typename E::From_t::Data_t, 1, sycl::access::mode::read> From_Acc_t;
        typedef sycl::accessor<typename E::To_t::Data_t, 1, sycl::access::mode::read> To_Acc_Read_t;
        typedef sycl::accessor<typename E::To_t::Data_t, 1, sycl::access::mode::write> To_Acc_Write_t;
        void advance(const Temporal_Param &tp, From_Acc_t& from_acc, To_Acc_t& to_acc)
        {
            static_cast<Derived*>(this)->advance(tp);
        }

        void synchronize(const Temporal_Param &tp, )
        {
            static_cast<Derived*>(this)->synchronize(tp);
        }
    };

    template <typename T>
    concept Link_type = Sycl_Graph::Invariant::Edge_type<T>;
    {
        E::Temporal_Param_t;
        E.advance(tp);
    };

    template <typename T>
    concept Network = Sycl_Graph::Invariant::Graph_type<T> && requires(T& graph)
    {
        graph.advance(tp);
        graph.read_state(tp);
        graph.read_node_states(tp);
        graph.reset();
        graph.terminate(x, tp);
        graph.simulate(Nt, tp);
    }

}

#endif