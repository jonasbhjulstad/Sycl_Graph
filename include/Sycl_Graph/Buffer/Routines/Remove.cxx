module;
#include <Sycl_Graph/Common/common.hpp>
export module Sycl.Buffer.Remove;
// removes elements at offset to offset+size
export template <typename T, std::unsigned_integral uI_t = uint32_t>
void buffer_remove(sycl::buffer<T, 1> &buf, sycl::queue &q, uI_t offset = 0, uI_t size = 0) {
  if (size == 0) {
    size = buf.size();
  }
  if constexpr (sizeof(T) > 0) {
    q.submit([&](sycl::handler &h) {
      auto acc = buf.template get_access<sycl::access::mode::write>(h);
      h.parallel_for(size, [=](sycl::id<1> i) { acc[i + offset] = acc[i + offset + size]; });
    });
  }
}

export template <typename... Ts> void buffer_remove(std::tuple<sycl::buffer<Ts, 1>...> &bufs,
                                             sycl::queue &q, uI_t offset = 0, uI_t size = 0) {
  (buffer_remove(std::get<sycl::buffer<Ts, 1>>(bufs), q, offset, size), ...);
}

export template <typename... Ts> void buffer_remove(std::tuple<sycl::buffer<Ts, 1>...> &bufs,
                                             sycl::queue &q, const std::vector<uI_t> &indices,
                                             uI_t N_max = std::numeric_limits<uI_t>::max()) {
  const auto buf_size = std::min<uI_t>(std::get<0>(bufs).size(), N_max);
  if (buf_size > 0 && indices.size() > 0) {
    auto indices_sorted = indices;
    std::sort(indices_sorted.begin(), indices_sorted.end());
    auto offset = indices_sorted[0];
    auto size = indices_sorted[indices_sorted.size() - 1] - offset + 1;
    buffer_remove(bufs, q, offset, size);
  }
}

export template <typename... Ts> void buffer_remove(std::tuple<sycl::buffer<Ts, 1>...> &bufs,
                                             sycl::queue &q, const std::vector<uI_t> &indices) {}

export template <typename T, typename... Ts>
void buffer_remove(std::tuple<sycl::buffer<Ts, 1>...> &bufs, sycl::queue &q, auto condition) {
  auto indices = buffer_get_indices(std::get<sycl::buffer<T, 1>>(bufs), q, condition);
  buffer_remove(bufs, q, indices);
}