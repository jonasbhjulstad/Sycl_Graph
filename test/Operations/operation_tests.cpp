#include <Sycl_Graph/Algorithms/Operations/Degree_Properties.hpp>
#include <Sycl_Graph/Algorithms/Operations/Operations.hpp>
#include <Sycl_Graph/Buffer/Sycl/Edge_Buffer.hpp>
#include <Sycl_Graph/Buffer/Sycl/Vertex_Buffer.hpp>
#include <Sycl_Graph/Graph/Base/Graph.hpp>
#include <Sycl_Graph/Graph/Sycl/Graph.hpp>
#include <iostream>
typedef Vertex<float> fVertex_t;
typedef Vertex<int> iVertex_t;
typedef Vertex<double> dVertex_t;
typedef Edge<uint32_t> u32Edge_t;
typedef Edge<uint32_t, iVertex_t, fVertex_t> i_f_edge_t;
typedef Edge<uint32_t, fVertex_t, iVertex_t> f_i_edge_t;
typedef Edge<uint32_t, iVertex_t, iVertex_t> i_i_edge_t;

template <Edge_Buffer_type Edge_Buffer_t> struct Square_Extract_Op
    : public Edge_Extract_Operation<Edge_Buffer_t, Square_Extract_Op<Edge_Buffer_t>>

{
  using Base_t = Edge_Extract_Operation<Edge_Buffer_t, Square_Extract_Op<Edge_Buffer_t>>;
  using Base_t::Base_t;
  typedef uint32_t Target_t;
  static constexpr sycl::access_mode target_access_mode = sycl::access_mode::write;

  Square_Extract_Op() = default;
  Square_Extract_Op(const Edge_Buffer_t&) {}

  void invoke(const auto& edge_acc, const auto& from_acc, const auto& to_acc, auto& target_acc,
              sycl::handler& h) const {
    std::cout << "Extraction: Iterating over " << edge_acc.size() << " edges" << std::endl;
    std::cout << "Target buffer of size " << target_acc.size() << std::endl;
    sycl::stream os(1024, 256, h);
    h.parallel_for(target_acc.size(), [=](auto i) {
      auto data = edge_acc[i].data;
      os << "Extracting edge " << i << " with value " << data << sycl::endl;
      target_acc[i] = data * data;
    });
  }
  template <typename Graph_t> size_t target_buffer_size(const Graph_t& G) const {
    return G.template current_size<typename Base_t::Edge_t>();
  }
};

template <Edge_Buffer_type Edge_Buffer_t> struct Square_Transform_Op
    : public Transform_Operation<Square_Transform_Op<Edge_Buffer_t>> {
  typedef uint32_t Target_t;
  typedef uint32_t Source_t;
  using Base_t = Transform_Operation<Square_Transform_Op>;
  using Base_t::Base_t;
  static constexpr sycl::access_mode target_access_mode = sycl::access_mode::write;
  Square_Transform_Op() = default;
  Square_Transform_Op(const Edge_Buffer_t&) {}
  void invoke(const auto& source_acc, auto& target_acc, sycl::handler& h) const {
    std::cout << "Transform: Iterating over " << source_acc.size() << " elements" << std::endl;
    std::cout << "Source buffer of size " << source_acc.size() << std::endl;
    std::cout << "Target buffer of size " << target_acc.size() << std::endl;
    h.parallel_for(source_acc.size(), [=](auto i) {
      auto elem = source_acc[i];
      target_acc[i] = elem * elem;
    });
  }
  template <typename Graph_t> size_t target_buffer_size(const Graph_t& G) const {
    return G.template current_size<typename Edge_Buffer_t::Edge_t>();
  }
};
template <Edge_Buffer_type Edge_Buffer_t> struct Square_Inject_Op
    : public Edge_Inject_Operation<Edge_Buffer_t, Square_Inject_Op<Edge_Buffer_t>> {
  using Base_t = Edge_Inject_Operation<Edge_Buffer_t, Square_Inject_Op<Edge_Buffer_t>>;
  using Base_t::Base_t;
  typedef uint32_t Source_t;

  Square_Inject_Op() = default;
  Square_Inject_Op(const Edge_Buffer_t&) {}

  void invoke(auto& edge_acc, auto& from_acc, auto& to_acc, const auto& source_acc,
              sycl::handler& h) const {
    std::cout << "Injection: Iterating over " << source_acc.size() << " elements" << std::endl;
    std::cout << "Source buffer of size " << source_acc.size() << std::endl;
    assert(edge_acc.data.size() >= source_acc.size());
    h.parallel_for(source_acc.size(), [=](auto i) {
      auto elem = source_acc[i];
      edge_acc.data[i] = elem * elem;
    });
  }

  template <typename Graph_t> size_t source_buffer_size(const Graph_t& G) const {
    return G.template current_size<typename Base_t::Edge_t>();
  }
};

int main() {
  sycl::queue q(sycl::gpu_selector_v);

  std::vector<fVertex_t> fvertices
      = {fVertex_t{0, 0.0f}, fVertex_t{1, 1.0f}, fVertex_t{2, 2.0f}, fVertex_t{3, 3.0f}};
  std::vector<iVertex_t> ivertices
      = {iVertex_t{0, 0}, iVertex_t{1, 1}, iVertex_t{2, 2}, iVertex_t{3, 3}};

  std::vector<i_f_edge_t> i_f_edges
      = {i_f_edge_t(1, 1, 1),
         i_f_edge_t(2, 3, 2)};  // i_f_edge_t(1, 2), i_f_edge_t(2, 3), i_f_edge_t(3, 2)};
  std::vector<f_i_edge_t> f_i_edges
      = {f_i_edge_t(1, 2, 10),
         f_i_edge_t(2, 1, 10)};  // f_i_edge_t(1, 2), f_i_edge_t(2, 3), f_i_edge_t(3, 2)};
  std::vector<i_i_edge_t> i_i_edges
      = {i_i_edge_t(1, 2, 10),
         i_i_edge_t(1, 2, 10)};  // i_i_edge_t(1, 2), i_i_edge_t(2, 3), i_i_edge_t(3, 2)};

  Sycl::Vertex_Buffer fv_buf(q, fvertices);
  Sycl::Vertex_Buffer iv_buf(q, ivertices);
  Sycl::Edge_Buffer i_f_e_buf(q, i_f_edges);
  Sycl::Edge_Buffer f_i_e_buf(q, f_i_edges);
  Sycl::Edge_Buffer i_i_e_buf(q, i_i_edges);

  q.wait();
  Buffer_Pack vertex_buffer(fv_buf, iv_buf);
  Buffer_Pack edge_buffer(i_f_e_buf, f_i_e_buf, i_i_e_buf);
  Sycl::Graph graph(vertex_buffer, edge_buffer, q);

  std::cout << "Graph has " << graph.N_vertices() << " vertices and " << graph.N_edges()
            << " edges." << std::endl;

  Square_Extract_Op i_f_extract(i_f_e_buf);
  Square_Transform_Op f_transform(i_f_e_buf);
  Square_Inject_Op i_f_inject(i_f_e_buf);

  std::tuple<std::string, std::string, std::string> edge_type_names
      = std::make_tuple("i_f_extract", "float_transform", "i_f_inject");

  auto ops = std::make_tuple(i_f_extract, f_transform, i_f_inject);
  q.wait();

  auto i_f_target = Sycl::create_target_buffer(graph, i_f_extract);
  auto i_f_source = Sycl::create_source_buffer(graph, i_f_inject);

  assert(i_f_target->size() == i_f_source->size());
  assert(i_f_target->size() == i_f_edges.size());

  // get contents of i_f_edge_t buffer
  auto edge_prev = i_f_e_buf.get_edges();

  std::cout << "i_f_edge_t buffer contents: " << std::endl;

  for (auto& e : edge_prev) {
    std::cout << e.from << " -> " << e.to << ": " << e.data << std::endl;
  }

  auto ex_event = invoke_extraction(graph, i_f_extract, i_f_target);
  q.wait();
  std::cout << std::endl;
  auto transform_event = invoke_transform(graph, f_transform, i_f_target, i_f_source, ex_event);
  q.wait();
  auto inject_event = invoke_injection(graph, i_f_inject, i_f_source, transform_event);
  inject_event.wait();
  q.wait();
  auto source_res = buffer_get(*i_f_source, q);
  auto target_res = buffer_get(*i_f_target, q);
  q.wait();

  std::cout << "target buffer contents: " << std::endl;
  for (auto& e : target_res) {
    std::cout << e << ", ";
  }
  std::cout << "source buffer contents: " << std::endl;
  for (auto& e : source_res) {
    std::cout << e << ", ";
  }
  std::cout << std::endl;


  std::cout << "i_f_edge_t buffer contents: " << std::endl;
  auto edge_after = i_f_e_buf.get_edges();
  for (auto& e : edge_after) {
    std::cout << e.from << " -> " << e.to << ": " << e.data << std::endl;
  }
  return 0;
}