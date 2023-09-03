#ifndef SYCL_GRAPH_UTILS_MATH_HPP
#define SYCL_GRAPH_UTILS_MATH_HPP
#include <CL/sycl.hpp>
#include <cmath>
#include <cstdint>

namespace Sycl_Graph {
  SYCL_EXTERNAL std::size_t floor_div(std::size_t a, std::size_t b);

  // n choose k
  SYCL_EXTERNAL std::size_t n_choose_k(std::size_t n, std::size_t k);
}  // namespace Sycl_Graph

#endif
