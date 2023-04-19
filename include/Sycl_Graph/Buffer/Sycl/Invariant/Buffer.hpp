#ifndef SYCL_GRAPH_BUFFER_SYCL_INVARIANT_BUFFER_HPP
#define SYCL_GRAPH_BUFFER_SYCL_INVARIANT_BUFFER_HPP
#include <CL/sycl.hpp>
#include <Sycl_Graph/Buffer/Sycl/Base/Buffer.hpp>
#include <Sycl_Graph/Buffer/Invariant/Buffer.hpp>


namespace Sycl_Graph::Sycl::Invariant
{
    template <Sycl_Graph::Sycl::Base::Buffer_type ... Bs>
    struct Buffer: public Sycl_Graph::Invariant::Buffer<Bs ...>
    {

        using Base_t = Sycl_Graph::Invariant::Buffer<Bs ...>;
        typedef typename Base_t::uI_t uI_t;
        typedef typename Base_t::Data_t Data_t;

        using Base_t::Base_t;

        template <sycl::access_mode Mode, typename Buffer_t>
        auto get_access(sycl::handler &h)
        {   
            return this->template get_buffer<Buffer_t>().template get_access<Mode>(h);
        }

        template <sycl::access_mode Mode, typename Buffer_t, typename D = void>
        auto get_access(sycl::handler &h)
        {
            return this->template get_buffer<Buffer_t>().template get_access<Mode, D>(h);
        }

    };
}

#endif