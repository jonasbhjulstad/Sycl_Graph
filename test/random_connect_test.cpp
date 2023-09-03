#include <Sycl_Graph/Generation/Random_Connect.hpp>
#include <Sycl_Graph/Metrics/Edge_Limits.hpp>
#include <oneapi/dpl/random>
int main()
{
    using namespace Sycl_Graph;
    using RNG = oneapi::dpl::ranlux24;
    sycl::queue q;
    std::vector<uint32_t> from{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<uint32_t> to{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<uint32_t> seeds{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    auto N_edges_max = bipartite_graph_max_edges(from.size(), to.size());
    auto p = 0.5;
    auto edges = random_connect<oneapi::dpl::ranlux24>(q, from, to, seeds, p);
    p = 0.0;
    edges = random_connect<oneapi::dpl::ranlux24>(q, from, to, seeds, p);
    assert(edges.size() == 0);
    p = 1.0;
    edges = random_connect<oneapi::dpl::ranlux24>(q, from, to, seeds, p);
    assert(edges.size() == N_edges_max);

    return 0;
}
