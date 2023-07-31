#ifndef SYCL_GRAPH_BUFFER_SYCL_INVARIANT_BUFFER_HPP
#define SYCL_GRAPH_BUFFER_SYCL_INVARIANT_BUFFER_HPP
#include <CL/sycl.hpp>
#include <Sycl_Graph/Buffer/Sycl/Buffer.hpp>
#include <Sycl_Graph/Buffer/Base/Buffer_Pack.hpp>

namespace Sycl_Graph::Sycl
{
    template <Sycl_Graph::Sycl::Buffer_type ... Bs>
    struct Buffer_Pack: public Sycl_Graph::Buffer_Pack<Bs ...>
    {

        using Base_t = Sycl_Graph::Buffer_Pack<Bs ...>;
        typedef typename Base_t::Buffer_t Buffer_t;
        using Base_t::get_buffer;
        using Base_t::Base_t;

        template <sycl::access_mode Mode, typename Buffer_t>
        auto get_access(sycl::handler &h)
        {
            return this->template get_buffer<Buffer_t>()->template get_access<Mode>(h);
        }
    };

    template <typename T>
    concept Buffer_Pack_type = true;
    template <Sycl_Graph::Sycl::Buffer_type ... Bs>
    Buffer_Pack(Bs& ... bs) -> Buffer_Pack<Bs ...>;
}

#endif
