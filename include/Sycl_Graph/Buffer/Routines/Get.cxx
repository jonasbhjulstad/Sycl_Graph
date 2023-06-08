module;
#include <Sycl_Graph/Common/common.hpp>
 module Sycl.Buffer.Get;
export template <typename T, std::unsigned_integral uI_t = uint32_t>
std::vector<T> buffer_get(sycl::buffer<T> &buf, sycl::queue &q) {
  auto res = std::vector<T>(buf.size());
  auto res_buf = sycl::buffer<T, 1>(res.data(), sycl::range<1>(res.size()));
  q.submit([&](sycl::handler &h) {
     auto buf_acc = buf.export template get_access<sycl::access::mode::read>(h);
     auto res_acc = res_buf.export template get_access<sycl::access::mode::write>(h);
     h.parallel_for(buf.size(), [=](sycl::id<1> i) { res_acc[i] = buf_acc[i]; });
   }).wait();

  return res;
}

export template <typename T, std::unsigned_integral uI_t = uint32_t>
std::vector<T> buffer_get(sycl::buffer<T, 1> &buf, sycl::queue &q,
                                 const std::vector<uI_t> &indices) {
  auto condition = [&indices](auto i) {
    return std::find(indices.begin(), indices.end(), i) != indices.end();
  };
  return buffer_get(buf, q, condition);
}

export template <typename... Ts>
std::tuple<std::vector<Ts>...> buffer_get(std::tuple<sycl::buffer<Ts, 1>...> &bufs,
                                                 sycl::queue &q, uI_t offset = 0, uI_t size = 0) {
  return std::make_tuple(buffer_get(std::get<sycl::buffer<Ts, 1>>(bufs), q, offset, size)...);
}

export template <typename... Ts>
std::tuple<std::vector<Ts>...> buffer_get(std::tuple<sycl::buffer<Ts, 1>...> &bufs,
                                                 sycl::queue &q, const std::vector<uI_t> &indices) {
  return std::make_tuple(buffer_get(std::get<sycl::buffer<Ts, 1>>(bufs), q, indices)...);
}

export template <typename... Ts>
std::tuple<std::vector<Ts>...> buffer_get(std::tuple<sycl::buffer<Ts, 1>...> &bufs,
                                                        sycl::queue &q, auto condition) {
  return std::make_tuple(buffer_get(std::get<sycl::buffer<Ts, 1>>(bufs), q, condition)...);
}

export template <typename T>
 std::optional<std::vector<T>> buffer_get(std::shared_ptr<sycl::buffer<T>> buf) {
  return buf ? std::optional<std::vector<T>>(buffer_get(*buf)) : std::nullopt;
}

export template <typename... Buf_t> auto buffer_get(std::tuple<std::shared_ptr<Buf_t>...> &bufs) {
  return std::apply([&](auto &&...buf) { return std::make_tuple(buffer_get(buf)...); }, bufs);
}

export template <typename T, std::unsigned_integral uI_t = uint32_t>
 std::vector<uI_t> buffer_get_indices(sycl::buffer<T, 1> &buf, sycl::queue &q,
                                            bool (*condition)(uI_t)) {
  std::vector<uI_t> res(buf.size());
  sycl::buffer<uI_t, 1> res_buf(res.data(), sycl::range<1>(buf.size()));
  q.submit([&](sycl::handler &h) {
    auto acc = buf.export template get_access<sycl::access::mode::read>(h);
    auto res_acc = res_buf.export template get_access<sycl::access::mode::write>(h);
    h.parallel_for(buf.size(), [=](sycl::id<1> i) {
      if (condition(i)) {
        res_acc[i] = i;
      }
    });
  });
  std::remove_if(res.begin(), res.end(), [](auto i) { return i == 0; });
  return res;
}

export template <typename T, std::unsigned_integral uI_t = uint32_t>
 std::vector<uI_t> buffer_get_indices(sycl::buffer<T, 1> &buf, sycl::queue &q,
                                            const std::vector<T> &elements) {
  if (buf.size() > 0) {
    std::vector<uI_t> res(elements.size(), std::numeric_limits<uI_t>::max());
    sycl::buffer<T, 1> elements_buf(elements.data(), sycl::range<1>(elements.size()));
    sycl::buffer<uI_t, 1> res_buf(res.data(), sycl::range<1>(elements.size()));
    auto event = q.submit([&](sycl::handler &h) {
      auto acc = buf.export template get_access<sycl::access::mode::read>(h);
      auto elements_acc = elements_buf.export template get_access<sycl::access::mode::read>(h);
      auto res_acc = res_buf.export template get_access<sycl::access::mode::write>(h);
      h.parallel_for(buf.size(), [=](sycl::id<1> i) {
        for (auto j = 0; j < acc.size(); j++) {
          if (acc[j] == elements_acc[i]) {
            res_acc[j] = i;
            return;
          }
        }
      });
    });
    event.wait();
    // copy res_buf to res
    auto res_acc = res_buf.export template get_access<sycl::access::mode::read>();
    for (auto i = 0; i < res.size(); i++) {
      res[i] = res_acc[i];
    }
    std::erase_if(res, [](uI_t i) { return i == std::numeric_limits<uI_t>::max(); });

    return res;
  }
  return {};
}
