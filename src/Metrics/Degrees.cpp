#include <Sycl_Graph/Generation/Complete_Graph.hpp>
#include <Sycl_Graph/Metrics/Degrees.hpp>
#include <Sycl_Graph/Utils/Math.hpp>
#include <Sycl_Graph/Utils/work_groups.hpp>
namespace Sycl_Graph {
sycl::event compute_in_degrees(sycl::queue &q, Edgebuf_t<1> &edges,
                               sycl::buffer<uint32_t, 1> &degrees) {
  uint32_t N_vertices = degrees.get_range()[0];
  uint32_t N_edges = edges.get_range()[0];
  return q.submit([&](sycl::handler &h) {
    auto deg_acc = degrees.get_access<sycl::access::mode::write>(h);
    auto e_acc = edges.get_access<sycl::access::mode::read>(h);
    h.parallel_for(get_nd_range(q, N_vertices), [=](sycl::nd_item<1> it) {
      auto gid = it.get_global_id();
      deg_acc[gid] = 0;
      for (uint32_t e_idx = 0; e_idx < N_edges; ++e_idx) {
        deg_acc[gid] += (e_acc[e_idx].second == gid);
      }
    });
  });
}

sycl::event compute_out_degrees(sycl::queue &q, Edgebuf_t<1> &edges,
                                sycl::buffer<uint32_t, 1> &degrees) {
  uint32_t N_vertices = degrees.get_range()[0];
  uint32_t N_edges = edges.get_range()[0];
  return q.submit([&](sycl::handler &h) {
    auto deg_acc = degrees.get_access<sycl::access::mode::write>(h);
    auto e_acc = edges.get_access<sycl::access::mode::read>(h);
    h.parallel_for(get_nd_range(q, N_vertices), [=](sycl::nd_item<1> it) {
      auto gid = it.get_global_id();
      deg_acc[gid] = 0;
      for (uint32_t e_idx = 0; e_idx < N_edges; ++e_idx) {
        deg_acc[gid] += (e_acc[e_idx].first == gid);
      }
    });
  });
}

sycl::event compute_undirected_degrees(sycl::queue &q, Edgebuf_t<1> &edges,
                                       sycl::buffer<uint32_t, 1> &degrees) {
  uint32_t N_vertices = degrees.get_range()[0];
  uint32_t N_edges = edges.get_range()[0];
  return q.submit([&](sycl::handler &h) {
    auto deg_acc = degrees.get_access<sycl::access::mode::write>(h);
    auto e_acc = edges.get_access<sycl::access::mode::read>(h);
    h.parallel_for(get_nd_range(q, N_vertices), [=](sycl::nd_item<1> it) {
      auto gid = it.get_global_id();
      deg_acc[gid] = 0;
      for (uint32_t e_idx = 0; e_idx < N_edges; ++e_idx) {
        deg_acc[gid] += (e_acc[e_idx].first == gid) || (e_acc[e_idx].second == gid);
      }
    });
  });
}

sycl::event compute_undirected_affinity_degrees(sycl::queue &q, Edgebuf_t<1> &edges,
                                                sycl::buffer<uint32_t, 1> &vcm,
                                                uint32_t N_communities,
                                                sycl::buffer<uint32_t, 1> &degrees) {
  auto N_connections = degrees.get_range()[0];
  auto ccm_vec = complete_graph(N_communities, false, true);
  if (N_connections != ccm_vec.size()) {
    throw std::runtime_error("Degrees buffer size does not match the number of connections");
  }

  auto N_edges = edges.get_range()[0];

  auto ccm = Edgebuf_t<1>(ccm_vec.data(), sycl::range<1>(N_connections));

  return q.submit([&](sycl::handler &h) {
    auto deg_acc = degrees.get_access<sycl::access::mode::write>(h);
    auto e_acc = edges.get_access<sycl::access::mode::read>(h);
    auto vcm_acc = vcm.get_access<sycl::access::mode::read>(h);
    auto ccm_acc = ccm.get_access<sycl::access::mode::read>(h);
    h.parallel_for(get_nd_range(q, N_connections), [=](sycl::nd_item<1> it) {
        auto connection_ids = ccm_acc[it.get_global_id()];
        auto gid = it.get_global_id();
        deg_acc[gid] = 0;
        for (auto i = 0; i < N_edges; i++) {
          auto e = e_acc[i];
          deg_acc[gid] += (vcm_acc[e.first] == connection_ids.first)
                          && (vcm_acc[e.second] == connection_ids.second);
          deg_acc[gid] += (vcm_acc[e.first] == connection_ids.second)
                          && (vcm_acc[e.second] == connection_ids.first);
        }
    });
  });
}
}  // namespace Sycl_Graph
