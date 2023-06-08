module;
#include <Sycl_Graph/Common/common.hpp>
export module Sycl.Buffer.Combine;
export template <typename T, std::unsigned_integral uI_t = uint32_t>
sycl::buffer<T, 1> buffer_combine(sycl::queue &q, sycl::buffer<T, 1> buf0,
                                         sycl::buffer<T, 1> buf1, uI_t size0 = 0, uI_t size1 = 0) {
  if (size0 == 0) {
    size0 = buf0.size();
  }
  if (size1 == 0) {
    size1 = buf1.size();
  }
  sycl::buffer<T, 1> new_buf(size0 + size1);
  q.submit([&](sycl::handler &h) {
    auto acc0 = buf0.template get_access<sycl::access::mode::read>(h);
    auto new_acc = new_buf.template get_access<sycl::access::mode::write>(h);
    h.parallel_for(size0, [=](sycl::id<1> i) { new_acc[i] = acc0[i]; });
  });
  q.submit([&](sycl::handler &h) {
    auto acc1 = buf1.template get_access<sycl::access::mode::read>(h);
    auto new_acc = new_buf.template get_access<sycl::access::mode::write>(h);
    h.parallel_for(size1, [=](sycl::id<1> i) { new_acc[i + size0] = acc1[i]; });
  });
  return new_buf;
}

export template <std::unsigned_integral uI_t = uint32_t, typename... Ts>
sycl::buffer<Ts...> buffer_combine(sycl::queue &q, std::tuple<Ts...> bufs, uI_t size0 = 0,
                                          uI_t size1 = 0) {
  return std::apply([&](auto &...buf) { return buffer_combine(q, buf..., size0, size1); }, bufs);
}