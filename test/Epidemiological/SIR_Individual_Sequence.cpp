#include <Sycl_Graph/Algorithms/Generation/Base/Graph_Generation.hpp>
#include <Sycl_Graph/Buffer/Sycl/Edge_Buffer.hpp>
#include <Sycl_Graph/Buffer/Sycl/Vertex_Buffer.hpp>
#define EPIDEMIOLOGICAL_POPULATION_COUNT_DEBUG 1
#define OPERATION_DEBUG_TARGET_BUFS 1
// #define OPERATION_VERBOSE_DEBUG 1
#include <spdlog/spdlog.h>

#include <Sycl_Graph/Buffer/Sycl/Buffer_Routines.hpp>
#include <Sycl_Graph/Epidemiological/Epidemiological.hpp>
#include <Sycl_Graph/Graph/Base/Graph.hpp>
#include <Sycl_Graph/Graph/Sycl/Graph.hpp>
#include <iostream>
#include <random>
using namespace Sycl_Graph::Epidemiological;
using namespace Sycl_Graph::Sycl;

template <typename T> void population_print(sycl::queue& q, std::shared_ptr<sycl::buffer<T>> buf) {
  auto vec = Sycl_Graph::buffer_get(*buf, q);

  uint32_t S = 0;
  uint32_t I = 0;
  uint32_t R = 0;
  if constexpr (std::is_same_v<T, SIR_Individual_State_t>) {
    std::cout << "Graph State: \n";
    for (auto&& val : vec) {
      switch (val) {
        case SIR_INDIVIDUAL_S:
          S++;
          break;
        case SIR_INDIVIDUAL_I:
          I++;
          break;
        case SIR_INDIVIDUAL_R:
          R++;
          break;
        default:
          break;
      }
    }
  } else {
    std::cout << "Count State: \n";
    S = vec[0];
    I = vec[1];
    R = vec[2];
  }
  std::cout << "S: " << S << " I: " << I << " R: " << R << std::endl;
}

void print_target_buffers(sycl::queue& q, auto& bufs) {
  std::apply([&](auto&&... buf) {}, bufs);
}

void print_shared_ptr_use_count(auto& pp_tup) {
  auto print_count = [](auto&& ptr) { std::cout << ptr.use_count() << " "; };

  std::cout << "Shared_ptr use count: ";
  auto tuple_print_count
      = [&](auto&& p_tup) { std::apply([&](auto&&... ptr) { (print_count(ptr), ...); }, p_tup); };

  std::apply([&](auto&&... pp) { (tuple_print_count(pp), ...); }, pp_tup);

  std::cout << std::endl;
}
auto generate_nodes_edges(uint32_t N_pop, float p_ER, uint32_t seed) {
  std::vector<uint32_t> node_ids(N_pop);
  std::iota(node_ids.begin(), node_ids.end(), 0);
  std::vector<SIR_Individual_Edge_t> links
      = Sycl_Graph::random_connect<SIR_Individual_Edge_t>(node_ids, node_ids, p_ER, false, seed);
  // generate initial infections with p_I0
  float p_I0 = 0.1f;
  // initialize mt
  std::mt19937 mt(seed);
  std::bernoulli_distribution dist(p_I0);

  std::vector<SIR_Individual_Vertex_t> nodes(N_pop);
  for (auto&& node : nodes) {
    node.data = dist(mt) ? SIR_INDIVIDUAL_I : SIR_INDIVIDUAL_S;
  }
  auto p_nodes = std::make_shared<std::vector<SIR_Individual_Vertex_t>>(nodes);
  auto p_links = std::make_shared<std::vector<SIR_Individual_Edge_t>>(links);
  return std::make_tuple(nodes, links);
}


int main() {
  spdlog::set_level(spdlog::level::debug);
  sycl::queue q(sycl::gpu_selector_v);
  uint32_t N_pop = 100;
  uint32_t seed = 87;
  float p_ER = 1.0f;
  // get GPU work group size
  auto device = q.get_device();
  auto N_wg = device.get_info<sycl::info::device::max_work_group_size>();
  std::cout << "Max work group size: " << N_wg << std::endl;

  auto [nodes, links] = generate_nodes_edges(N_pop, p_ER, seed);

  float p_I = 1e-1f;
  float p_R = 1e-1f;

  std::for_each(links.begin(), links.end(), [&](auto& edge) { edge.data = p_I; });

  std::cout << "Number of links: " << links.size() << std::endl;

  {
    SIR_Individual_Edge_Buffer_t e_buf = make_edge_buffer(q, links);
    SIR_Individual_Vertex_Buffer_t v_buf = make_vertex_buffer(q, nodes);

    Sycl_Graph::Buffer_Pack vertex_buffer(v_buf);
    Sycl_Graph::Buffer_Pack edge_buffer(e_buf);
    Sycl_Graph::Sycl::Graph graph(vertex_buffer, edge_buffer, q);

    std::cout << "Graph has " << graph.N_vertices() << " vertices and " << graph.N_edges()
              << " edges." << std::endl;
    q.wait();

    {
        auto [ops, buffers] = create_inplace_SIR_step(graph, p_I, p_R, N_wg);

    }

  }
  return 0;
}
