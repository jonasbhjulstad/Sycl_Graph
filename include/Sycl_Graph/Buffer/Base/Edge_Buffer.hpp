#ifndef SYCL_GRAPH_EDGE_BUFFER_HPP
#define SYCL_GRAPH_EDGE_BUFFER_HPP
#include <Sycl_Graph/Buffer/Base/Buffer.hpp>
#include <Sycl_Graph/Graph/Base/Graph_Types.hpp>
namespace Sycl_Graph
{


    template <typename T>
    concept Edge_Buffer_type = true;

}

#endif
