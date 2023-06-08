  module;
  #include <Sycl_Graph/Common/common.hpp>
  export module Sycl.Buffer.Assign;
  export template <std::unsigned_integral uI_t = uint32_t, typename... Ts>
  void buffer_assign(std::tuple<sycl::buffer<Ts, 1>...> &bufs, sycl::queue &q,
                     const std::vector<uI_t> &indices, const std::tuple<std::vector<Ts>...> &vecs) {
    const auto buf_size = std::get<0>(bufs).size();
    auto src_bufs = std::apply(
        [&](const auto &...vecs) {
          return std::make_tuple(
              (sycl::buffer<Ts, 1>(vecs.data(), sycl::range<1>(vecs.size())))...);
        },
        vecs);
    sycl::buffer<uI_t, 1> indices_buf(indices.data(), sycl::range<1>(indices.size()));
    q.submit([&](sycl::handler &h) {
      auto dest_accs = std::make_tuple(
          std::get<sycl::buffer<Ts, 1>>(bufs).template get_access<sycl::access::mode::read_write>(
              h)...);
      auto src_accs = std::make_tuple(
          std::get<sycl::buffer<Ts, 1>>(src_bufs).template get_access<sycl::access::mode::read>(
              h)...);
      auto indices_acc = indices_buf.template get_access<sycl::access::mode::read>(h);
      h.parallel_for(buf_size, [=](sycl::id<1> i) {
        if (i == indices_acc[i]) {
          std::apply(
              [&](auto &...dest_acc) {
                std::apply([&](auto &...src_acc) { ((dest_acc[i] = src_acc[i]), ...); }, src_accs);
              },
              dest_accs);
        }
      });
    });
  }

  export template <typename Target_t, std::unsigned_integral uI_t = uint32_t, typename... Ts>
  void buffer_assign(const std::vector<Target_t> &target, std::tuple<sycl::buffer<Ts, 1>...> &bufs,
                     sycl::queue &q, const std::vector<Ts> &...vecs) {
    sycl::buffer<Target_t, 1> target_buf(target.data(), sycl::range<1>(target.size()));
    auto indices = buffer_get_indices(target_buf, q, [&](auto t) { return t == target; });
    buffer_assign(bufs, q, indices, vecs...);
  }

  export template <std::unsigned_integral uI_t = uint32_t, typename... Ts>
  uI_t buffer_assign_add(std::tuple<sycl::buffer<Ts, 1>...> &bufs, sycl::queue &q,
                         const std::vector<uI_t> &indices,
                         const std::tuple<std::vector<Ts>...> &vecs,
                         uI_t N_max = std::numeric_limits<uI_t>::max()) {
    const auto buf_size = std::min<uI_t>(std::get<0>(bufs).size(), N_max);
    if (buf_size > 0 && indices.size() > 0) {
      buffer_assign(bufs, q, indices, vecs);
      buffer_print(bufs, q);
      std::tuple<std::vector<Ts>...> filtered_vecs;
      ((std::get<std::vector<Ts>>(filtered_vecs).reserve(buf_size - indices.size())), ...);
      for (auto i = 0; i < buf_size - indices.size(); i++) {
        if (std::find(indices.begin(), indices.end(), i) == indices.end()) {
          ((std::get<std::vector<Ts>>(filtered_vecs).push_back(std::get<std::vector<Ts>>(vecs)[i])),
           ...);
        }
      }
      buffer_add(bufs, filtered_vecs, q, buf_size);
      return std::get<0>(filtered_vecs).size();

    } else {
      buffer_add(bufs, vecs, q, buf_size);
      return std::get<0>(vecs).size();
    }
  }

  export template <typename T>
  sycl::event buffer_fill(sycl::buffer<T> &buf, const T &value, sycl::queue &q) {
    return q.submit([&](sycl::handler &h) {
      auto acc = buf.template get_access<sycl::access::mode::write>(h);
      h.fill(acc, value);
    });
  }

  export template <typename Target_t, std::unsigned_integral uI_t = uint32_t, typename... Ts>
  uI_t buffer_assign_add(std::tuple<sycl::buffer<Ts, 1>...> &bufs, sycl::queue &q,
                         const std::tuple<std::vector<Ts>...> &vecs,
                         uI_t N_max = std::numeric_limits<uI_t>::max()) {
    const auto &target = std::get<std::vector<Target_t>>(vecs);
    std::vector<uI_t> indices;
    if (std::get<0>(bufs).size() > 0) {
      indices = buffer_get_indices(std::get<sycl::buffer<Target_t, 1>>(bufs), q, target);
    }
    // auto target_buffers = tuple_filter<sycl::buffer<Bs, 1> ...>::template filter<sycl::buffer<Ts,
    // 1> ...>(bufs);
    return buffer_assign_add<uI_t, Ts...>(bufs, q, indices, vecs, N_max);
  }