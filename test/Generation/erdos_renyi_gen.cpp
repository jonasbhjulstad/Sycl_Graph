#include <Sycl_Graph/Graph/Sycl/Graph.hpp>
#include <Sycl_Graph/Network/SIR_Bernoulli/SIR_Bernoulli.hpp>
#include <Sycl_Graph/Graph/Graph_Generation.hpp>
#include <Static_RNG/distributions.hpp>
#include <Sycl_Graph/Math/math.hpp>
int main()
{

    using namespace Sycl_Graph::Dynamic::Network_Models;
    using namespace Sycl_Graph::Sycl::Network_Models;
    size_t N_pop = 100;
    float p_ER = 0.1;

    Static_RNG::default_rng rng;
    SIR_Graph G(100, 1000);
    SIR_Bernoulli_Network sir(G, 0.1, 0.1, rng);
    //generate sir_param
    size_t Nt = 100;
    std::vector<SIR_Bernoulli_Temporal_Param<float>> sir_param(Nt);
    std::generate(sir_param.begin(), sir_param.end(), [&]() {
        return SIR_Bernoulli_Temporal_Param<float>{0.1, 0.1, 100, 10};
    });

    generate_erdos_renyi(G, N_pop, p_ER, SIR_INDIVIDUAL_S, rng);

    sir.initialize();
    auto traj = sir.simulate(sir_param,Nt);
    auto traj_T = Sycl_Graph::transpose(traj);
    //print traj
    for (auto &x : traj_T) {
        for (auto &y : x) {
            std::cout << y << ",";
        }
        std::cout << std::endl;
    }
    //create random number generator
    //create SIR_bernoulli network



}