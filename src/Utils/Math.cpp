#include <Sycl_Graph/Utils/Math.hpp>

namespace Sycl_Graph {
  SYCL_EXTERNAL std::size_t floor_div(std::size_t a, std::size_t b) {
    return static_cast<std::size_t>(std::floor(static_cast<double>(a) / static_cast<double>(b)));
  }

  SYCL_EXTERNAL std::size_t n_choose_k(std::size_t n, std::size_t k) {
    if (k > n) return 0;
    if (k * 2 > n) k = n - k;
    if (k == 0) return 1;

    std::size_t result = n;
    for (std::size_t i = 2; i <= k; ++i) {
      result *= (n - i + 1);
      result /= i;
    }
    return result;
  }
}  // namespace Sycl_Graph