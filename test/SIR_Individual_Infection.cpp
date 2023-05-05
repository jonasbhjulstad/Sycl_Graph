
#include <Sycl_Graph/Epidemiological/SIR_Individual/
#include <Sycl_Graph/Algorithms/Operations/Operations.hpp>
#include <Sycl_Graph/Buffer/Sycl/Edge_Buffer.hpp>
#include <Sycl_Graph/Buffer/Sycl/Vertex_Buffer.hpp>
#include <Sycl_Graph/Graph/Base/Graph.hpp>
#include <Sycl_Graph/Graph/Sycl/Graph.hpp>
#include <iostream>
using namespace Sycl_Graph;

int main() {
  sycl::queue q(sycl::gpu_selector_v);



  q.wait();
  auto ids = i_f_e_buf.get_valid_ids();
  std::cout << "Valid ids" << std::endl;
  for (const auto& id : ids) {
    std::cout << id.from << "," << id.to << std::endl;
  }

  Sycl_Graph::Buffer_Pack vertex_buffer(fv_buf, iv_buf);
  Sycl_Graph::Buffer_Pack edge_buffer(i_f_e_buf, f_i_e_buf, i_i_e_buf);
  Sycl_Graph::Sycl::Graph graph(vertex_buffer, edge_buffer, q);

  std::cout << "Graph has " << graph.N_vertices() << " vertices and " << graph.N_edges()
            << " edges." << std::endl;

  Sycl_Graph::Sycl::Directed_Vertex_Degree_Op i_f_op(iv_buf, fv_buf, i_f_e_buf);
  Sycl_Graph::Sycl::Directed_Vertex_Degree_Op f_i_op(fv_buf, iv_buf, f_i_e_buf);
  Sycl_Graph::Sycl::Directed_Vertex_Degree_Op i_i_op(iv_buf, iv_buf, i_i_e_buf);
  std::tuple<std::string, std::string, std::string> edge_type_names
      = std::make_tuple("integer-float", "float-integer", "integer-integer");

  // auto ops = std::make_tuple(i_f_op_in, i_f_op_out);//, f_i_op_in, f_i_op_out);
  auto ops = std::make_tuple(i_f_op, f_i_op, i_i_op);
  q.wait();
  auto properties = Sycl_Graph::Sycl::apply_single_operations(graph, ops);
  auto printvecpair = [&](const auto& vec, const std::string name) {
    std::cout << "op for " << name << " edges" << std::endl;
    for (const auto& v : vec) {
      std::cout << "From: " << v.from << ", To: " << v.to << std::endl;
    }
    std::cout << std::endl;
  };

  auto print_vec = [&](const auto& vec, const auto& name) {
    std::cout << "op for " << name << " edges" << std::endl;
    for (const auto& v : vec) {
      std::cout << v << ",";
    }
    std::cout << std::endl;
  };
  uint32_t n = 0;

  std::cout << "Degrees" << std::endl;

  std::apply(
      [&](auto&&... name) {
        std::apply(
            [&](auto&&... args) {
              // ((std::cout << "Property " << n++ << std::endl), printvecpair(args), ...);
              (print_vec(args, name), ...);
            },
            properties);
      },
      edge_type_names);

  // Sycl_Graph::Sycl::Undirected_Vertex_Degree_Op i_f_op_undirected(iv_buf, fv_buf, i_f_e_buf);
  // Sycl_Graph::Sycl::Undirected_Vertex_Degree_Op f_i_op_undirected(fv_buf, iv_buf, f_i_e_buf);
  // Sycl_Graph::Sycl::Undirected_Vertex_Degree_Op i_i_op_undirected(iv_buf, iv_buf, i_i_e_buf);

  // auto ops_undirected = std::make_tuple(i_f_op_undirected, f_i_op_undirected, i_i_op_undirected);

  // auto properties_undirected = Sycl_Graph::Sycl::apply_single_operations(graph, ops_undirected);
  // auto printvecpair_undirected = [&](const auto& vec, const std::string name) {
  //     std::cout << "op for " << name << " edges" << std::endl;
  //     for (const auto& v : vec) {
  //         std::cout << v << std::endl;
  //     }
  //     std::cout << std::endl;
  // };

  // std::apply([&](auto&&... name){
  // std::apply([&](auto&&... args) {
  //     // ((std::cout << "Property " << n++ << std::endl), printvecpair(args), ...);
  //     (printvecpair_undirected(args, name), ...);
  // }, properties_undirected);}, edge_type_names);

  return 0;
}