#ifndef SYCL_GRAPH_BUFFER_BASE_HPP
#define SYCL_GRAPH_BUFFER_BASE_HPP
#include <Sycl_Graph/type_helpers.hpp>

namespace Sycl_Graph
{
    template <typename T>
    concept Buffer_type = true;
    
}

#endif