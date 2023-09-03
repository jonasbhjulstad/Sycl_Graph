#include <Sycl_Graph/Generation/SBM.hpp>

sycl::event random_connect(sycl::queue &q, sycl::buffer<uint32_t, 1> &from, sycl::buffer<uint32_t, 1> &to, sycl::buffer<RNG> &rngs, const float p, Edgebuf_t<1> &edges, sycl::buffer<uint32_t> N_edges_tot)


template <typename RNG>
sycl::event generate_SBM(sycl::queue& q, const std::vector<std::vector<uint32_t>>& vertex_ids, const std::vector<std::vector<float>>& p_SBM, sycl::buffer<RNG>& rngs, )
