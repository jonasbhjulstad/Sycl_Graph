module;
#include <Sycl_Graph/Common/common.hpp>
export module Sycl.Buffer.Copy template <typename T>
export void buffer_copy(sycl::buffer<T, 1> &buf, sycl::queue &q, const std::vector<T> &vec) {
  if (vec.size() == 0) {
    return;
  }
  sycl::buffer<T, 1> tmp_buf(vec.data(), sycl::range<1>(vec.size()));
  q.submit([&](sycl::handler &h) {
    auto acc = buf.template get_access<sycl::access::mode::write>(h);
    auto tmp_acc = tmp_buf.template get_access<sycl::access::mode::read>(h);
    h.parallel_for(vec.size(), [=](sycl::id<1> i) { acc[i] = tmp_acc[i]; });
  });
}