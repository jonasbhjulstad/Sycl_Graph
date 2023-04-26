#ifndef SYCL_GRAPH_VERTEX_BUFFER_BASE_HPP
#define SYCL_GRAPH_VERTEX_BUFFER_BASE_HPP
#include <Sycl_Graph/Buffer/Base/Buffer.hpp>
namespace Sycl_Graph
{
    template <typename T>
    concept Vertex_Buffer_type = true;
}
#endif