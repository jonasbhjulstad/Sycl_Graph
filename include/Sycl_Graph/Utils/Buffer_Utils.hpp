#ifndef SYCL_GRAPH_BUFFER_UTILS_HPP
#define SYCL_GRAPH_BUFFER_UTILS_HPP
#include <Sycl_Graph/Common.hpp>

template <typename T>
sycl::event read_buffer(sycl::queue& q, sycl::buffer<T>& buf, std::vector<T>& result, bool exact = false)
{
    if (exact && result.size() != buf.get_count())
    {
        throw std::runtime_error("read_buffer: result vector size does not match buffer size");
    }
    auto N = std::min<std::size_t>({result.size(), buf.size()});
    return q.submit([&](sycl::handler& h)
    {
        auto acc = sycl::accessor<T, 1, sycl::access::mode::read>(buf, h, sycl::range<1>(N));
        h.copy(acc, result.data());
    });
}




#endif
