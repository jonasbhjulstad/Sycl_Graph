#include <Sycl_Graph/Generation/SBM.hpp>
#include <Sycl_Graph/Metrics/Edge_Limits.hpp>
#include <oneapi/dpl/random>
#include <Sycl_Graph/Utils/zip.hpp>
int main()
{
    using namespace Sycl_Graph;
    using namespace Sycl_Graph::USM;
    using RNG = oneapi::dpl::ranlux24;
    sycl::queue q;

    auto N_pop = 100;
    auto N_communities = 10;
    float p_in = 0.5;
    float p_out = 0.5;
    auto seed = 23;
    auto N_rngs = 16;
    auto graph = generate_planted_SBM<RNG>(q, N_pop, N_communities, p_in, p_out, seed, N_rngs);

    assert(graph.get_N_communities() == N_communities);
    assert(graph.get_N_vertices() == N_pop*N_communities);

    auto sbm_complete = generate_planted_SBM<RNG>(q, N_pop, N_communities, 1.0f, 1.0f, seed);

    auto sbm_complete_edges = sbm_complete.get_flat_edges();
    std::sort(sbm_complete_edges.begin(), sbm_complete_edges.end());

    auto G = complete_graph(N_pop*N_communities, false, true);

    auto zip_edges = zip(graph.get_flat_edges(), G);
    assert(std::all_of(zip_edges.begin(), zip_edges.end(), [&](auto e){
        auto e_0 = std::get<0>(e);
        auto e_1 = std::get<1>(e);
        return e_0 == e_1;
    }));

    return 0;
}
