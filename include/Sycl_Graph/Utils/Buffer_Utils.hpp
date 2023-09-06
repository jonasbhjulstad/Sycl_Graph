#ifndef SYCL_GRAPH_BUFFER_UTILS_HPP
#define SYCL_GRAPH_BUFFER_UTILS_HPP
#include <Sycl_Graph/Common.hpp>
#include <algorithm>
#include <numeric>
#include <vector>
namespace Sycl_Graph {
  template <typename T> sycl::event read_buffer(sycl::queue& q, sycl::buffer<T>& buf,
                                                std::vector<T>& result, bool exact = false) {
    if (exact && result.size() != buf.get_count()) {
      throw std::runtime_error("read_buffer: result vector size does not match buffer size");
    }
    auto N = std::min<std::size_t>({result.size(), buf.size()});
    return q.submit([&](sycl::handler& h) {
      auto acc = sycl::accessor<T, 1, sycl::access::mode::read>(buf, h, sycl::range<1>(N));
      h.copy(acc, result.data());
    });
  }
}  // namespace Sycl_Graph
namespace Sycl_Graph::USM {

#ifdef BUFFER_UTIL_DEBUG
  static int shared_usm_init_counter = 0;
#endif

  template <typename T> struct usm_deleter {
    usm_deleter(sycl::queue& q) : q_(q) {}
    void operator()(T* p) {
#ifdef BUFFER_UTIL_DEBUG
      static int N_inst = 0;
      std::cout << "usm_deleter: " << shared_usm_init_counter-- << std::endl;
#endif
      sycl::free(p, q_);
    }

  private:
    sycl::queue& q_;
  };

  template <typename T>
  auto initialize_device_usm(const std::vector<T>& data, sycl::queue& q, sycl::event& event) {
    auto p = sycl::malloc_device<T>(data.size(), q);
    event = q.submit([&](sycl::handler& h) { h.copy(data.data(), p, data.size()); });
    return p;
  }

  template <typename T> auto make_shared_usm(sycl::queue& q, std::size_t N) {
    auto p = sycl::malloc_shared<T>(N, q);
#ifdef BUFFER_UTIL_DEBUG
    std::cout << "make_shared_usm: " << shared_usm_init_counter++ << " with size " << N
              << std::endl;
#endif
    return std::shared_ptr<T>(p, usm_deleter<T>(q));
  };

  template <typename T> auto make_shared_usm(sycl::queue& q, const std::vector<T>& v) {
#ifdef BUFFER_UTIL_DEBUG

    std::cout << "make_shared_usm: " << shared_usm_init_counter++ << " with size " << v.size()
              << std::endl;
#endif
    auto p = sycl::malloc_shared<T>(v.size(), q);
    std::copy(v.begin(), v.end(), p);
    return std::shared_ptr<T>(p, usm_deleter<T>(q));
  };

  template <typename T> inline auto shared_offset(const std::shared_ptr<T>& p, std::size_t offset) {
    return std::shared_ptr<T>(p, p.get() + offset);
  };

  template <typename T>
  inline auto shared_usm_accumulate(std::shared_ptr<T>& p, std::size_t N, auto f = std::plus<>()) {
    std::vector<T> data(p.get(), p.get() + N);
    return std::accumulate(data.begin(), data.end(), T{}, f);
  }

  template <typename T> inline auto shared_usm_partial_sum(std::shared_ptr<T>& p, std::size_t N) {
    std::vector<T> data(p.get(), p.get() + N);
    std::vector<T> result(data.size(), T{});
    std::partial_sum(data.begin(), data.end(), result.begin());
    return result;
  }

  template <typename T>
  inline auto read_device_usm(std::vector<T>& result, T* p_usm, sycl::queue& q) {
    return q.submit([&](sycl::handler& h) { h.copy(p_usm, result.data(), result.size()); });
  }

}  // namespace Sycl_Graph::USM

#endif
