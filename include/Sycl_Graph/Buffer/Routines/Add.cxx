  module;
  #include <Sycl_Graph/Common/common.hpp>
  export module Sycl.Buffer.Add;
  export template <typename T>
  void buffer_add(sycl::buffer<T, 1> &dest_buf, sycl::buffer<T, 1> src_buf, sycl::queue &q,
                  uint32_t offset = 0) {
    if (src_buf.size() == 0) {
      return;
    }
    if constexpr (sizeof(T) > 0) {
      if (dest_buf.size() < src_buf.size() + offset) {
        buffer_resize(dest_buf, q, src_buf.size() + offset);
      }

      q.submit([&](sycl::handler &h) {
        auto src_acc = src_buf.template get_access<sycl::access::mode::read>(h);
        auto dest_acc = dest_buf.template get_access<sycl::access::mode::write>(h);
        h.parallel_for(src_buf.size(), [=](sycl::id<1> i) { dest_acc[i + offset] = src_acc[i]; });
      });
    }
  }

  export template < typename... Ts>
  void buffer_add(std::tuple<sycl::buffer<Ts, 1>...> &dest_bufs,
                  std::tuple<sycl::buffer<Ts, 1>...> &src_bufs, sycl::queue &q, uint32_t offset = 0) {
    auto bufsize = std::get<0>(dest_bufs).size();
    static_assert(std::conjunction_v<std::bool_constant<(sizeof(Ts) != 0)>...>,
                  "All buffer types must have non-zero size.");
    if (bufsize == 0) {
      return;
    }

    if (bufsize < offset + std::get<0>(src_bufs).size()) {
      buffer_resize(dest_bufs, q, bufsize + std::get<0>(src_bufs).size());
    }

    q.submit([&](sycl::handler &h) {
      auto dest_accs = std::apply(
          [&](auto &...dest_bufs) {
            return std::make_tuple(dest_bufs.template get_access<sycl::access::mode::write>(h)...);
          },
          dest_bufs);
      auto src_accs = std::apply(
          [&](auto &...src_bufs) {
            return std::make_tuple(src_bufs.template get_access<sycl::access::mode::read>(h)...);
          },
          src_bufs);
      h.parallel_for(std::get<0>(src_bufs).size(), [=](sycl::id<1> i) {
        std::apply(
            [&](auto &...dest_acc) {
              std::apply([&](auto &...src_acc) { ((dest_acc[i + offset] = src_acc[i]), ...); },
                         src_accs);
            },
            dest_accs);
      });
    });
  }

  export template < typename... Ts>
  void buffer_add(std::tuple<sycl::buffer<Ts, 1>...> &dest_bufs,
                  const std::tuple<std::vector<Ts>...> &src_vecs, sycl::queue &q, uint32_t offset = 0) {
    // create buffers for src_vecs
    auto src_bufs = std::apply(
        [&](auto &...src_vecs) {
          return std::make_tuple(
              sycl::buffer<Ts, 1>(src_vecs.data(), sycl::range<1>(src_vecs.size()))...);
        },
        src_vecs);
    buffer_add(dest_bufs, src_bufs, q, offset);
  }

  export template <typename T>
  void buffer_add(std::vector<sycl::buffer<T, 1> &> &bufs,
                         const std::vector<const std::vector<T> &> &vecs, sycl::queue &q,
                         const std::vector<uint32_t> &offsets) {
    for (uint32_t i = 0; i < vecs.size(); ++i) {
      buffer_add(bufs[i], vecs[i], q, offsets[i]);
    }
  }

  export template <typename T>
  void buffer_add(sycl::buffer<T> &buf, const std::vector<T> &data, sycl::queue &q,
                  uint32_t offset = 0) {
    sycl::buffer<T> tmp_buf(data.data(), sycl::range<1>(data.size()));
    buffer_add(buf, tmp_buf, q, offset);
  }

  export template <typename T>
  std::vector<T> buffer_get(sycl::buffer<T> &buf) {
    auto buf_acc = buf.get_host_access();
    std::vector<T> res(buf.size());
    for (int i = 0; i < buf.size(); ++i) {
      res[i] = buf_acc[i];
    }
    return res;
  }