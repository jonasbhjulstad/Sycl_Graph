#ifndef SYCL_GRAPH_GRAPH_HPP
#define SYCL_GRAPH_GRAPH_HPP
#include <Sycl_Graph/Common.hpp>
using Edge_t = std::pair<uint32_t, uint32_t>;
template <std::size_t N = 1>
using Edgebuf_t = sycl::buffer<Edge_t, N>;

#endif
