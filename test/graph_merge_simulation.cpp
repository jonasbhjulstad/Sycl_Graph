
#include <Sycl_Graph/Graph/Base/Graph.hpp>
#include <Sycl_Graph/Graph/Graph_Generation.hpp>
#include <Sycl_Graph/Math/math.hpp>
#include <Sycl_Graph/Network/SIR_Metapopulation/SIR_Metapopulation.hpp>
#include <Sycl_Graph/path_config.hpp>
#include <Static_RNG/distributions.hpp>
#include <algorithm>
#include <filesystem>

using namespace Static_RNG::;
static constexpr size_t NV = 100;
std::vector<uint32_t> N_pop = std::vector<uint32_t>(NV, 1000);
std::vector<normal_distribution<float>> I0(N_pop.size());
std::vector<normal_distribution<float>> R0(N_pop.size());
std::vector<float> alpha(N_pop.size(), 0.01);
std::vector<float> node_beta(N_pop.size(), 0.1);
std::vector<float> edge_beta(N_pop.size(), 0.1);
int main()
{


  std::transform(N_pop.begin(), N_pop.end(), I0.begin(), [](auto x)
                 { return normal_distribution<float>(x * 0.1, x * 0.01); });

  using namespace Sycl_Graph::Sycl::Network_Models;
  using Sycl_Graph_Dynamic::Network_Models::generate_erdos_renyi;
  using namespace Sycl_Graph::Network_Models;
  float p_ER = 0.5;
  //create profiling queue
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::enable_profiling{});
  int seed = 777;
  Static_RNG::default_rng rng;
  const std::vector<uint32_t> G0_ids = Sycl_Graph::range(0, NV);
  const std::vector<uint32_t> G1_ids = Sycl_Graph::range(NV, 2 * NV);
  auto G0 = generate_erdos_renyi<SIR_Metapopulation_Graph>(q, NV, p_ER, G0_ids);
  auto G1 = generate_erdos_renyi<SIR_Metapopulation_Graph>(q, NV, p_ER, G1_ids);

  std::cout << G0.N_vertices() << std::endl;
  std::cout << G1.N_vertices() << std::endl;
  // std::cout << G.N_vertices() << std::endl;
  // std::cout << G.N_edges() << std::endl;

  // std::filesystem::create_directory(Sycl_Graph_Sycl_GRAPH_DATA_DIR + std::string("/Edgelists"));

  // auto G = Sycl_Graph_Dynamic::Network_Models::random_connect(G, G0_ids, G1_ids, p_ER, rng);

  // G.write_edgelist(Sycl_Graph_Sycl_GRAPH_DATA_DIR + std::string("/Edgelists/merge_graph.csv"));
  

}