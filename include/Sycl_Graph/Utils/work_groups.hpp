#ifndef SYCL_GRAPH_UTILS_WORK_GROUPS_HPP
#define SYCL_GRAPH_UTILS_WORK_GROUPS_HPP

#include <Sycl_Graph/Common_Types.hpp>
namespace Sycl_Graph {
  inline auto get_wg_size(sycl::queue& q) {
    auto device = q.get_device();
    return device.get_info<sycl::info::device::max_work_group_size>();
  }

  inline auto get_compute_size(sycl::queue& q) {
    auto device = q.get_device();
    auto max_wg = device.get_info<sycl::info::device::max_work_group_size>();
  }

  sycl::nd_range<1> get_nd_range(sycl::queue& q, std::size_t N_threads) {
    auto device = q.get_device();
    auto max_wg = device.get_info<sycl::info::device::max_work_group_size>();
    auto max_cu = device.get_info<sycl::info::device::max_compute_units>();
    auto N_wg = std::min<std::size_t>({N_threads, max_wg});
    auto N_compute = std::min<std::size_t>({N_threads, max_wg * max_cu});
    return sycl::nd_range<1>(sycl::range<1>(N_compute), sycl::range<1>(N_wg));
  }

  auto get_N_per_work_item(std::size_t N, sycl::nd_range<1> nd_range) {
    return static_cast<uint32_t>(
        std::ceil(static_cast<double>(N) / nd_range.get_global_range()[0]));
  }
}  // namespace Sycl_Graph
#endif
