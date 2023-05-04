#ifndef SYCL_GRAPH_ALGORITHMS_PROPERTIES_EDGE_OPERATIONS_HPP
#define SYCL_GRAPH_ALGORITHMS_PROPERTIES_EDGE_OPERATIONS_HPP
#include <Sycl_Graph/Algorithms/Operations/Operation_Types.hpp>
namespace Sycl_Graph::Sycl {

  template <Graph_type Graph_t, Operation_type Op>
  sycl::event vertex_extraction(Graph_t& graph, sycl::queue& q, const Op& operation,
                                sycl::buffer<typename Op::Result_t>& result_buf,
                                sycl::event dep_event = {}) {
    using Vertex_t = typename Op::Vertex_t;
    return q.submit([&](sycl::handler& h) {
      h.depends_on(dep_event);
      auto result_acc = result_buf.template get_access<Op::result_access_mode>(h);
      auto vertex_acc = graph.template get_vertex_access<sycl::access_mode::read, Vertex_t>(h);
      operation(vertex_acc, result_acc, h);
    });
  }

  template <Graph_type Graph_t, Operation_type Op>
  sycl::event vertex_injection(Graph_t& graph, sycl::queue& q, const Op& operation,
                               sycl::buffer<typename Op::Source_t>& source_buf,
                               sycl::event dep_event = {}) {
    using Vertex_t = typename Op::Vertex_t;
    return q.submit([&](sycl::handler& h) {
      h.depends_on(dep_event);
      auto source_acc = source_buf.template get_access<sycl::access::mode::read>(h);
      auto vertex_acc
          = graph.template get_vertex_access<Op::vertex_access_mode, Vertex_t>(h);
      operation(source_acc, vertex_acc, h);
    });
  }

  template <Graph_type Graph_t, Operation_type Op>
  sycl::event vertex_modification(Graph_t& graph, sycl::queue& q, const Op& operation,
                                  sycl::event dep_event = {}) {
    using Vertex_t = typename Op::Vertex_t;
    return q.submit([&](sycl::handler& h) {
      h.depends_on(dep_event);
      auto vertex_acc
          = graph.template get_vertex_access<Op::vertex_access_mode, Vertex_t>(h);
      operation(source_acc, vertex_acc, h);
    });
  }

  template <Sycl_Graph::Vertex_Buffer_type Vertex_Buffer_t> struct Vertex_Inject_Op {
    typedef typename Vertex_Buffer_t::Vertex_t::Data_t Source_t;
    static constexpr Operation_Target_t operation_target = Operation_Target_Vertex;
    static constexpr Operation_Type_t operation_type = Operation_Modify_Vertices;
    static constexpr sycl::access::mode vertex_access_mode = sycl::access::mode::write;
    typedef typename Vertex_Buffer_t::Vertex_t Vertex_t;
    Vertex_Inject_Op(const std::tuple<Vertex_Buffer_t>& buf) {}

    void operator()(const auto& source_acc, auto& v_acc, sycl::handler& h) const {
      h.parallel_for(v_acc.get_count(), [=](auto i) { v_acc[i].data = source_acc[i]; });
    }
    template <Graph_type Graph_t> size_t result_buffer_size(const Graph_t& G) const {
      return G.vertex_buf.template get_buffer<Vertex_Buffer_t>().current_size();
    }
  };

}  // namespace Sycl_Graph::Sycl
#endif