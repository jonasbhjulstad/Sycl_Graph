#ifndef SYCL_GRAPH_ALGORITHMS_PROPERTIES_INVARIANT_OPERATIONS_HPP
#define SYCL_GRAPH_ALGORITHMS_PROPERTIES_INVARIANT_OPERATIONS_HPP
#include <Sycl_Graph/Algorithms/Operations/Operations.hpp>
namespace Sycl_Graph::Sycl
{



    template <Operation_type... Op>
    struct Invariant_Edge_Op
    {
        std::tuple<Op...> operations;
        template <tuple_like ... Buffer_Packs>
        Invariant_Edge_Op(){}
    };
}

#endif