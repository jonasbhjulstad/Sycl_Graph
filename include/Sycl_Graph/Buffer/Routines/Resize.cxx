module;
#include <Sycl_Graph/Common/common.hpp>
export module Sycl.Buffer.Resize;
export template <typename T>
void buffer_resize(sycl::buffer<T, 1> &buf, sycl::queue &q, size_t new_size) {
  if (buf.size() == new_size) {
    return;
  }
  sycl::buffer<T, 1> new_buf(new_size);
  auto smallest_buf_size = std::min<size_t>(buf.size(), new_size);
  q.submit([&](sycl::handler &h) {
    auto acc = buf.template get_access<sycl::access::mode::read>(h);
    auto new_acc = new_buf.template get_access<sycl::access::mode::write>(h);
    h.parallel_for(smallest_buf_size, [=](sycl::id<1> i) { new_acc[i] = acc[i]; });
  });
  buf = new_buf;
}

export template <typename... Ts> void buffer_resize(std::tuple<sycl::buffer<Ts, 1>...> &bufs,
                                                    sycl::queue &q, size_t new_size) {
  std::tuple<sycl::buffer<Ts, 1>...> new_bufs = std::make_tuple(sycl::buffer<Ts, 1>(new_size)...);
  q.submit([&](sycl::handler &h) {
    auto src_accs = std::make_tuple(
        std::get<sycl::buffer<Ts, 1>>(bufs).template get_access<sycl::access::mode::read>(h)...);
    auto dest_accs = std::make_tuple(
        std::get<sycl::buffer<Ts, 1>>(new_bufs).template get_access<sycl::access::mode::write>(
            h)...);
    h.parallel_for(new_size, [=](sycl::id<1> i) {
      std::apply(
          [&](auto &...dest_acc) {
            std::apply([&](auto &...src_acc) { ((dest_acc[i] = src_acc[i]), ...); }, src_accs);
          },
          dest_accs);
    });
  });

  bufs = new_bufs;
}