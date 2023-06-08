
import Sycl.Graph;
import Sycl.Buffer.Vertex;
import Sycl.Buffer.Edge;
import Epidemiological;
import Base.Graph.Generation;
#include <iostream>
#include <random>

auto generate_nodes_edges(uint32_t N_pop, float p_ER, uint32_t seed) {
  std::vector<uint32_t> node_ids(N_pop);
  std::iota(node_ids.begin(), node_ids.end(), 0);
  std::vector<SIR_Individual_Infection_Edge_t> links
      = random_connect<SIR_Individual_Infection_Edge_t>(node_ids, node_ids, p_ER, false, seed);
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
  uint32_t N_pop = 1000;
  uint32_t seed = 87;
  float p_ER = 1.0f;
  auto [nodes, links] = generate_nodes_edges(N_pop, p_ER, seed);

  float p_I = 1e-3f;
  float p_R = 1e-3f;

  std::for_each(links.begin(), links.end(), [&](auto& edge) { edge.data = p_I; });

  SIR_Individual_Edge_Buffer_t e_buf(q, links);
  SIR_Individual_Vertex_Buffer_t v_buf(q, nodes);

  Buffer_Pack vertex_buffer(v_buf);
  Buffer_Pack edge_buffer(e_buf);
  Sycl::Graph graph(vertex_buffer, edge_buffer, q);

  std::cout << "Graph has " << graph.N_vertices() << " vertices and " << graph.N_edges()
            << " edges." << std::endl;

  SIR_Individual_Recovery_Op recovery_op(v_buf, p_R);
  auto edge_seeds = generate_seed_buf(graph.N_edges(), seed);
  SIR_Individual_Infection_Op infection_op(e_buf, v_buf, edge_seeds, p_I);
  SIR_Individual_Population_Count_Extract_Op initial_pop_count;
  SIR_Individual_Population_Count_Transform_Op pop_count_op;
  auto initial_pop_buf = create_target_buffer(graph, initial_pop_count);
  auto rec_pop_buf = create_target_buffer(graph, pop_count_op);
  auto inf_pop_buf = create_target_buffer(graph, pop_count_op);

  auto rec_buf = create_target_buffer(graph, recovery_op);
  auto inf_buf = create_target_buffer(graph, infection_op);

  q.wait();

  auto initial_pop_event = invoke_extraction(graph, initial_pop_count, initial_pop_buf);
  auto rec_event = invoke_extraction(graph, recovery_op, rec_buf, initial_pop_event);
  auto rec_pop_event = invoke_transform(graph, pop_count_op, rec_buf, rec_pop_buf, rec_event);

  auto inf_event = invoke_transform(graph, infection_op, rec_buf, inf_buf, rec_pop_event);
  auto inf_pop_event = invoke_transform(graph, pop_count_op, inf_buf, inf_pop_buf, inf_event);

  inf_pop_event.wait();

  auto init_pop = buffer_get(initial_pop_buf);
  auto rec_pop = buffer_get(rec_pop_buf);
  auto inf_pop = buffer_get(inf_pop_buf);
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