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
  auto vec = buffer_get(*buf, q);

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

void population_print(sycl::queue& q, std::shared_ptr<void> buf) {}

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
  std::vector<std::pair<uint32_t, uint32_t>> links
      = Sycl_Graph::random_connect(node_ids, node_ids, p_ER, false, seed);
  // generate initial infections with p_I0
  float p_I0 = 0.1f;
  // initialize mt
  std::mt19937 mt(seed);
  std::bernoulli_distribution dist(p_I0);

  std::vector<SIR_Individual_State_t> nodes(N_pop);
  std::generate(nodes.begin(), nodes.end(),
                [&]() { return dist(mt) ? SIR_INDIVIDUAL_I : SIR_INDIVIDUAL_S; });
  return std::make_tuple(nodes, links);
}

template <typename... Acc_Ts, typename Derived>
sycl::event invoke_operation(sycl::queue& q, Operation_Base<Derived, Acc_Ts...>& op, auto&& bufs,
                             sycl::event dep_event = {}) {
  return q.submit([&](sycl::handler& h) {
    h.depends_on(dep_event);
    auto accs = std::apply(
        [&](auto&&... buf) {
          return std::make_tuple(buf->template get_access<Acc_Ts::mode>(h)...);
        },
        bufs);
    std::apply([&](auto&&... acc) { op.__invoke(h, acc...); }, accs);
  });
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

  std::vector<float> link_data(links.size(), p_I);
  std::cout << "Number of links: " << links.size() << std::endl;


  {
    auto [node_buf, node_event] = buffer_create(q, nodes);
    auto [link_id_buf, link_id_event] = buffer_create(q, links);
    auto [link_data_buf, link_data_event] = buffer_create(q, link_data);
    q.wait();

    {
      SIR_Individual_Population_Count vertex_count_op(N_pop);
      SIR_Individual_Population_Count state_count_op(N_pop);
      SIR_Individual_Recovery recovery_op(0.1, N_wg);
      SIR_Individual_Infection infection_op(0.1, N_wg, N_pop);
      SIR_State_Injection inject_op{};

      auto ops = std::make_tuple(vertex_count_op, recovery_op, state_count_op, infection_op,
                                 state_count_op);

      auto [seeds, seed_gen_event] = Sycl_Graph::Sycl::generate_seed_buf(q, N_wg, seed);
      q.wait();

      auto [vertex_rec_buf, vrec_create_event] = buffer_create(q, nodes);
      auto [vertex_inf_buf, vinf_create_event] = buffer_create(q, nodes);
      std::vector<uint32_t> zero_buf = {0,0,0};

      auto [init_count_buf, init_count_create_event] = buffer_create(q, zero_buf);
      auto [rec_count_buf, rec_count_create_event] = buffer_create(q, zero_buf);
      auto [inf_count_buf, inf_count_create_event] = buffer_create(q, zero_buf);
      // auto node_buf = graph.template get_buffer<SIR_Individual_Vertex_t>();
      // auto e_buf = graph.template get_buffer<SIR_Individual_Edge_t>();
      // Source buffers
      auto init_count_source = node_buf;
      auto rec_source = node_buf;
      auto rec_count_source = vertex_rec_buf;
      auto inf_source = vertex_rec_buf;
      auto inf_count_source = vertex_inf_buf;
      auto inject_source = vertex_inf_buf;

      // Targets
      auto init_count_target = init_count_buf;
      auto rec_target = vertex_rec_buf;
      auto rec_count_target = rec_count_buf;
      auto inf_target = vertex_inf_buf;
      auto inf_count_target = inf_count_buf;
      auto inject_target = node_buf;

      q.wait();

      auto init_count_event = invoke_operation(
          q, vertex_count_op, std::make_tuple(init_count_source, init_count_target), sycl::event{});
      auto rec_event = invoke_operation(
          q, recovery_op, std::make_tuple(rec_source, rec_target, seeds), init_count_event);
      auto rec_count_event = invoke_operation(
          q, state_count_op, std::make_tuple(rec_count_source, rec_count_target), rec_event);
      auto buffer_copy_event = buffer_copy(q, rec_target, inf_target, rec_event);
      auto inf_event = invoke_operation(
          q, infection_op, std::make_tuple(link_id_buf, link_data_buf, inf_source, inf_target, seeds), buffer_copy_event);
      auto inf_count_event = invoke_operation(
          q, state_count_op, std::make_tuple(inf_count_source, inf_count_target), inf_event);
      auto inject_event = invoke_operation(
          q, inject_op, std::make_tuple(inject_source, inject_target), inf_count_event);

      auto events = std::make_tuple(init_count_event, rec_event, rec_count_event, inf_event,
                                    inf_count_event, inject_event);

      auto pop_bufs = std::make_tuple(init_count_buf, vertex_rec_buf, rec_count_buf, vertex_inf_buf,
                                      inf_count_buf, node_buf);

      std::apply([&](auto&... ev) { (ev.wait(), ...); }, events);

      q.wait();

      std::apply([&](auto&&... pop_buf) { (population_print(q, pop_buf), ...); }, pop_bufs);
    }
  }
  return 0;
}
