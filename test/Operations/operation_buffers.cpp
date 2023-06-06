#include <Sycl_Graph/Algorithms/Operations/Degree_Properties.hpp>
#include <Sycl_Graph/Algorithms/Operations/Operations.hpp>
#include <Sycl_Graph/Buffer/Sycl/Edge_Buffer.hpp>
#include <Sycl_Graph/Buffer/Sycl/Vertex_Buffer.hpp>
#include <Sycl_Graph/Graph/Base/Graph.hpp>
#include <Sycl_Graph/Graph/Sycl/Graph.hpp>
#include <iostream>
using namespace Sycl_Graph;
typedef Sycl_Graph::Vertex<float> fVertex_t;
typedef Sycl_Graph::Vertex<int> iVertex_t;
typedef Sycl_Graph::Vertex<double> dVertex_t;
typedef Sycl_Graph::Edge<float> fEdge_t;
typedef Sycl_Graph::Edge<fEdge_t, iVertex_t, fVertex_t> i_f_edge_t;
typedef Sycl_Graph::Edge<fEdge_t, fVertex_t, iVertex_t> f_i_edge_t;
typedef Sycl_Graph::Edge<fEdge_t, iVertex_t, iVertex_t> i_i_edge_t;

int main() {
  sycl::queue q(sycl::gpu_selector_v);

  std::vector<fVertex_t> fvertices
      = {fVertex_t{0, 0.0f}, fVertex_t{1, 1.0f}, fVertex_t{2, 2.0f}, fVertex_t{3, 3.0f}};
  std::vector<iVertex_t> ivertices
      = {iVertex_t{0, 0}, iVertex_t{1, 1}, iVertex_t{2, 2}, iVertex_t{3, 3}};

  std::vector<i_f_edge_t> i_f_edges
      = {i_f_edge_t(1, 2),
         i_f_edge_t(3, 2)};  // i_f_edge_t(1, 2), i_f_edge_t(2, 3), i_f_edge_t(3, 2)};
  std::vector<f_i_edge_t> f_i_edges
      = {f_i_edge_t(1, 2),
         f_i_edge_t(2, 1)};  // f_i_edge_t(1, 2), f_i_edge_t(2, 3), f_i_edge_t(3, 2)};
  std::vector<i_i_edge_t> i_i_edges
      = {i_i_edge_t(1, 2),
         i_i_edge_t(1, 2)};  // i_i_edge_t(1, 2), i_i_edge_t(2, 3), i_i_edge_t(3, 2)};

  Sycl_Graph::Sycl::Vertex_Buffer fv_buf(q, fvertices);
  Sycl_Graph::Sycl::Vertex_Buffer iv_buf(q, ivertices);
  Sycl_Graph::Sycl::Edge_Buffer i_f_e_buf(q, i_f_edges);
  Sycl_Graph::Sycl::Edge_Buffer f_i_e_buf(q, f_i_edges);
  Sycl_Graph::Sycl::Edge_Buffer i_i_e_buf(q, i_i_edges);

  q.wait();
  auto ids = i_f_e_buf.get_valid_ids();
  std::cout << "Valid ids" << std::endl;
  for (const auto& id : ids) {
    std::cout << id.from << "," << id.to << std::endl;
  }

  Sycl_Graph::Buffer_Pack vertex_buffer(fv_buf, iv_buf);
  Sycl_Graph::Buffer_Pack edge_buffer(i_f_e_buf, f_i_e_buf, i_i_e_buf);
  Sycl_Graph::Sycl::Graph graph(vertex_buffer, edge_buffer, q);

  Sycl_Graph::Sycl::Directed_Vertex_Degree_Op i_f_op(i_f_e_buf);
  Sycl_Graph::Sycl::Directed_Vertex_Degree_Op f_i_op(f_i_e_buf);
  Sycl_Graph::Sycl::Directed_Vertex_Degree_Op i_i_op(i_i_e_buf);

  auto ops = std::make_tuple(i_f_op, f_i_op, i_i_op);
  auto bufs = Sycl_Graph::Sycl::create_operation_buffers(graph, ops);

  return 0;
}