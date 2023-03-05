#ifndef SYCL_GRAPH_BUFFER_BASE_HPP
#define SYCL_GRAPH_BUFFER_BASE_HPP
#include <Sycl_Graph/type_helpers.hpp>

namespace Sycl_Graph::Base
{
    template <typename T>
    concept Buffer_type = requires(T buf)
    {
        typename T::Data_t;
        typename T::uI_t;
    };
    
}

#endif