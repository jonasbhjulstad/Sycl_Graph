#ifndef SYCL_GRAPH_COMMON_TYPES_HPP
#define SYCL_GRAPH_COMMON_TYPES_HPP
#include <Sycl_Graph/Common.hpp>

namespace Sycl_Graph
{

struct h1D_ranges
{
    h1D_ranges(std::size_t N_compute, std::size_t N_work_group): compute(N_compute), work_group(N_work_group) {}
    h1D_ranges() : compute(1), work_group(1) {}
    sycl::range<1> compute;
    sycl::range<1> work_group;
};
template <typename T, std::size_t N = 1>
using read_accessor = sycl::accessor<T, N, sycl::access::mode::read>;
template <typename T, std::size_t N = 1>
using write_accessor = sycl::accessor<T, N, sycl::access::mode::write>;
template <typename T, std::size_t N = 1>
using accessor = sycl::accessor<T, N, sycl::access::mode::read_write>;
}

#endif
