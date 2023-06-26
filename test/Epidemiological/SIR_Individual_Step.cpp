#include <Sycl_Graph/Buffer/Sycl/Edge_Buffer.hpp>
#include <Sycl_Graph/Buffer/Sycl/Vertex_Buffer.hpp>
#include <Sycl_Graph/Algorithms/Generation/Base/Graph_Generation.hpp>
#define EPIDEMIOLOGICAL_POPULATION_COUNT_DEBUG 1
#define OPERATION_DEBUG_TARGET_BUFS 1
// #define OPERATION_VERBOSE_DEBUG 1
#include <Sycl_Graph/Epidemiological/Epidemiological.hpp>
#include <Sycl_Graph/Graph/Base/Graph.hpp>
#include <Sycl_Graph/Graph/Sycl/Graph.hpp>
#include <iostream>
#include <random>

using namespace Sycl_Graph::Epidemiological;
using namespace Sycl_Graph::Sycl;

auto generate_nodes_edges(uint32_t N_pop, float p_ER, uint32_t seed) {
  std::vector<uint32_t> node_ids(N_pop);
  std::iota(node_ids.begin(), node_ids.end(), 0);
  std::vector<SIR_Individual_Infection_Edge_t> links
      = Sycl_Graph::random_connect<SIR_Individual_Infection_Edge_t>(node_ids, node_ids, p_ER, false, seed);
  // generate initial infections with p_I0
  float p_I0 = 0.1f;
  // initialize mt
  std::mt19937 mt(seed);
  std::bernoulli_distribution dist(p_I0);

  std::vector<SIR_Individual_Vertex_t> nodes(N_pop);
    for (auto &&node : nodes) {
      node.data = dist(mt) ? SIR_INDIVIDUAL_I : SIR_INDIVIDUAL_S;
    }
  return std::make_tuple(nodes, links);
}

auto generate_seed_buf(uint32_t N, uint32_t seed) {
  std::vector<uint32_t> vec(N);
  std::iota(vec.begin(), vec.end(), 0);
  std::mt19937 mt(seed);
  std::generate(vec.begin(), vec.end(), [&]() { return mt(); });
  sycl::buffer<uint32_t> buf(vec.data(), sycl::range<1>(N));
  return buf;
}

int main() {
  sycl::queue q(sycl::gpu_selector_v);
  uint32_t N_pop = 100;
  uint32_t seed = 87;
  float p_ER = 1.0f;
  //get GPU work group size
  auto device = q.get_device();
  auto N_wg = device.get_info<sycl::info::device::max_work_group_size>();
  std::cout << "Max work group size: " << N_wg << std::endl;

  auto [nodes, links] = generate_nodes_edges(N_pop, p_ER, seed);

  float p_I = 1.f;
  float p_R = 1e-1f;

  std::for_each(links.begin(), links.end(), [&](auto& edge) { edge.data = p_I; });

  std::cout << "Number of links: " << links.size() << std::endl;

  SIR_Individual_Edge_Buffer_t e_buf(q, links);
  SIR_Individual_Vertex_Buffer_t v_buf(q, nodes);

  Sycl_Graph::Buffer_Pack vertex_buffer(v_buf);
  Sycl_Graph::Buffer_Pack edge_buffer(e_buf);
  Sycl_Graph::Sycl::Graph graph(vertex_buffer, edge_buffer, q);

  std::cout << "Graph has " << graph.N_vertices() << " vertices and " << graph.N_edges()
            << " edges." << std::endl;

  //generate seeds equal to number of edges
  auto p_seeds = generate_seed_buf(N_wg, seed);

  SIR_Individual_Population_Count_Extract_Op initial_pop_count;
  SIR_Individual_Recovery_Op recovery_op(v_buf, p_R);
  SIR_Individual_Population_Count_Transform_Op pop_count_op;
  SIR_Individual_Infection_Op infection_op(e_buf, v_buf, p_I);
  auto edge_seeds = generate_seed_buf(graph.N_edges(), seed);
  auto p_edge_seeds = std::make_tuple(std::make_shared<sycl::buffer<uint32_t>>(edge_seeds));
  auto custom_buffers = std::make_tuple(std::tuple<>{}, p_edge_seeds, std::tuple<>{}, p_edge_seeds);
  auto initial_pop_buf = create_target_buffer(graph, initial_pop_count);
  auto rec_pop_buf = create_target_buffer(graph, pop_count_op);
  auto inf_pop_buf = create_target_buffer(graph, pop_count_op);


  auto init_pop = Sycl_Graph::buffer_get(initial_pop_buf);
  auto rec_pop = Sycl_Graph::buffer_get(rec_pop_buf);
  auto inf_pop = Sycl_Graph::buffer_get(inf_pop_buf);
  q.wait();

  auto rec_buf = create_target_buffer(graph, recovery_op);
  auto inf_buf = create_target_buffer(graph, infection_op);

  q.wait();

  auto op_tuple = std::make_tuple(initial_pop_count, recovery_op, pop_count_op, infection_op);

  auto [source_bufs, target_bufs] = create_operation_buffer_sequence(graph, op_tuple);



  auto events = invoke_operation_sequence(graph, op_tuple, source_bufs, target_bufs, custom_buffers);

  // auto initial_pop_event = invoke_extraction(graph, initial_pop_count, initial_pop_buf);
  // auto rec_event = invoke_extraction(graph, recovery_op, rec_buf, initial_pop_event);

  // auto rec_pop_event = invoke_transform(graph, pop_count_op, rec_b5uf, rec_pop_buf, rec_event);

  // rec_pop_event.wait();

  std::apply([&](auto&... events) { (events.wait(), ...); }, events);

  // auto inf_event = invoke_transform(graph, infection_op, rec_buf, inf_buf, rec_pop_event);
  // auto inf_pop_event = invoke_transform(graph, pop_count_op, inf_buf, inf_pop_buf, inf_event);

  // inf_pop_event.wait();

  q.wait();
  init_pop = Sycl_Graph::buffer_get(initial_pop_buf);
  rec_pop = Sycl_Graph::buffer_get(rec_pop_buf);
  inf_pop = Sycl_Graph::buffer_get(inf_pop_buf);
  q.wait();
  // print pop bufs
  std::cout << "Initial population: " << std::endl;
  for (auto& pop : *init_pop) {
    std::cout << pop << " ";
  }
  std::cout << std::endl;

  std::cout << "Recovered population: " << std::endl;
  for (auto& pop : *rec_pop) {
    std::cout << pop << " ";
  }

  std::cout << std::endl;

  std::cout << "Infected population: " << std::endl;
  for (auto& pop : *inf_pop) {
    std::cout << pop << " ";
  }
  std::cout << std::endl;

  return 0;
}
