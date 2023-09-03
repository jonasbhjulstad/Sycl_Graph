#ifndef SYCL_GRAPH_UTILS_WORK_GROUPS_HPP
#define SYCL_GRAPH_UTILS_WORK_GROUPS_HPP

#include <Sycl_Graph/Common_Types.hpp>

sycl::nd_range<1> get_nd_range(sycl::queue& q, std::size_t N_threads) {
  auto device = q.get_device();
  auto max_wg = device.get_info<sycl::info::device::max_work_group_size>();
  auto max_cu = device.get_info<sycl::info::device::max_compute_units>();
  auto N_compute = static_cast<std::size_t>(
      std::ceil(static_cast<double>(N_threads) / static_cast<double>(max_wg)));
  auto N_wg = std::min<std::size_t>({N_threads, max_wg});
  return sycl::nd_range<1>(sycl::range<1>(N_compute), sycl::range<1>(N_wg));
}

auto get_N_per_work_item(std::size_t N, sycl::nd_range<1> nd_range) {
  return static_cast<uint32_t>(std::ceil(
      static_cast<double>(N) / nd_range.get_global_range()[0]));
}

#endif
