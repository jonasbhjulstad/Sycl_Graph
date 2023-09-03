#include <Sycl_Graph/Generation/Random_Connect/Random_Connect_Buf_Inl.hpp>

template sycl::event Sycl_Graph::random_connect<oneapi::dpl::ranlux24>(
    sycl::queue &q, sycl::buffer<uint32_t, 1> &from, sycl::buffer<uint32_t, 1> &to,
    sycl::buffer<oneapi::dpl::ranlux24> &rngs, const float p, Edgebuf_t<1> &edges,
    sycl::buffer<uint32_t>& N_edges_tot);
template sycl::event Sycl_Graph::random_connect<oneapi::dpl::ranlux48>(
    sycl::queue &q, sycl::buffer<uint32_t, 1> &from, sycl::buffer<uint32_t, 1> &to,
    sycl::buffer<oneapi::dpl::ranlux48> &rngs, const float p, Edgebuf_t<1> &edges,
    sycl::buffer<uint32_t>& N_edges_tot);
template sycl::event Sycl_Graph::random_connect<oneapi::dpl::minstd_rand>(
    sycl::queue &q, sycl::buffer<uint32_t, 1> &from, sycl::buffer<uint32_t, 1> &to,
    sycl::buffer<oneapi::dpl::minstd_rand> &rngs, const float p, Edgebuf_t<1> &edges,
    sycl::buffer<uint32_t>& N_edges_tot);
template sycl::event Sycl_Graph::random_connect<oneapi::dpl::minstd_rand0>(
    sycl::queue &q, sycl::buffer<uint32_t, 1> &from, sycl::buffer<uint32_t, 1> &to,
    sycl::buffer<oneapi::dpl::minstd_rand0> &rngs, const float p, Edgebuf_t<1> &edges,
    sycl::buffer<uint32_t>& N_edges_tot);

template std::vector<Edge_t> Sycl_Graph::random_connect<oneapi::dpl::ranlux24>(
    sycl::queue &q, const std::vector<uint32_t> &from, const std::vector<uint32_t> &to,
    const std::vector<uint32_t> &seeds, const float p);
template std::vector<Edge_t> Sycl_Graph::random_connect<oneapi::dpl::ranlux48>(
    sycl::queue &q, const std::vector<uint32_t> &from, const std::vector<uint32_t> &to,
    const std::vector<uint32_t> &seeds, const float p);
template std::vector<Edge_t> Sycl_Graph::random_connect<oneapi::dpl::minstd_rand>(
    sycl::queue &q, const std::vector<uint32_t> &from, const std::vector<uint32_t> &to,
    const std::vector<uint32_t> &seeds, const float p);
template std::vector<Edge_t> Sycl_Graph::random_connect<oneapi::dpl::minstd_rand0>(
    sycl::queue &q, const std::vector<uint32_t> &from, const std::vector<uint32_t> &to,
    const std::vector<uint32_t> &seeds, const float p);

template std::vector<Edge_t> Sycl_Graph::self_connect<oneapi::dpl::ranlux24>(
    sycl::queue &q, const std::vector<uint32_t> ids, const std::vector<uint32_t> &seeds,
    const float p, bool self_loops);
template std::vector<Edge_t> Sycl_Graph::self_connect<oneapi::dpl::ranlux48>(
    sycl::queue &q, const std::vector<uint32_t> ids, const std::vector<uint32_t> &seeds,
    const float p, bool self_loops);
template std::vector<Edge_t> Sycl_Graph::self_connect<oneapi::dpl::minstd_rand>(
    sycl::queue &q, const std::vector<uint32_t> ids, const std::vector<uint32_t> &seeds,
    const float p, bool self_loops);
template std::vector<Edge_t> Sycl_Graph::self_connect<oneapi::dpl::minstd_rand0>(
    sycl::queue &q, const std::vector<uint32_t> ids, const std::vector<uint32_t> &seeds,
    const float p, bool self_loops);
