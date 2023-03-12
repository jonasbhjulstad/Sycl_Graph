#ifndef SYCL_GRAPH_INVARIANT_GRAPH_TYPES_HPP
#define SYCL_GRAPH_INVARIANT_GRAPH_TYPES_HPP
#include <concepts>
#include <Sycl_Graph/Graph/Base/Graph_Types.hpp>

namespace Sycl_Graph::Invariant
{

    template <typename D, typename _ID_t = uint32_t, _ID_t _invalid_id = std::numeric_limits<_ID_t>::max()>
    using Vertex = Sycl_Graph::Base::Vertex<D, _ID_t, _invalid_id>;

    template <typename T>
    concept Vertex_type = Sycl_Graph::Base::Vertex_type<T>;

    template <Sycl_Graph::Base::Edge_type E, Vertex_type _To, Vertex_type _From>
    struct Edge: public E
    {
        typedef typename E::ID_t ID_t;
        typedef typename E::Data_t Data_t;
        typedef _From From_t;
        typedef _To To_t;
        Edge(typename E::ID_t to, typename E::ID_t from): E(to, from) {}
        
    };


    template <typename T>
    concept Edge_type = Sycl_Graph::Base::Edge_type<T> && Sycl_Graph::Base::Vertex_type<typename T::To_t> && Sycl_Graph::Base::Vertex_type<typename T::From_t>;
}
#endif