#ifndef SYCL_GRAPH_ALGORITHMS_PROPERTIES_SYCL_DEGREE_PROPERTIES_HPP
#define SYCL_GRAPH_ALGORITHMS_PROPERTIES_SYCL_DEGREE_PROPERTIES_HPP
#include <CL/sycl.hpp>
#include <Sycl_Graph/Algorithms/Properties/Operation_Types.hpp>
#include <concepts>
namespace Sycl_Graph::Sycl {
  template <Sycl_Graph::Vertex_Buffer_type Vertex_Buffer_From_t,
            Sycl_Graph::Vertex_Buffer_type Vertex_Buffer_To_t,
            Sycl_Graph::Edge_Buffer_type Edge_Buffer_t>
  struct Directed_Vertex_Degree_Op {
    struct Result_t {
      uint32_t from = 0;
      uint32_t to = 0;
    };
    static constexpr Operation_Target_t operation_target = Operation_Target_Edge;
    static constexpr Operation_Type_t operation_type = Operation_Direct_Transform;
    static constexpr sycl::access_mode result_access_mode = sycl::access::mode::atomic;
    typedef typename Edge_Buffer_t::Edge_t Edge_t;
    typedef typename Vertex_Buffer_From_t::Vertex_t From_t;
    typedef typename Vertex_Buffer_To_t::Vertex_t To_t;

    Directed_Vertex_Degree_Op() = default;
    Directed_Vertex_Degree_Op(const Vertex_Buffer_From_t&, const Vertex_Buffer_To_t&,
                              const Edge_Buffer_t&) {}

    void operator()(const auto& edge_acc, const auto& from_acc, const auto& to_acc,
                    auto& result_acc, sycl::handler& h) {
      h.parallel_for(edge_acc.size(), [=](sycl::id<1> id) {
        auto& to_id = edge_acc[id].to;
        auto& from_id = edge_acc[id].from;
        if (to_id == Edge_t::invalid_id || from_id == Edge_t::invalid_id) return;
        result_acc[to_id].to++;
        result_acc[from_id].from++;
      });
    }
  };

  template <Sycl_Graph::Vertex_Buffer_type Vertex_Buffer_From_t,
            Sycl_Graph::Vertex_Buffer_type Vertex_Buffer_To_t,
            Sycl_Graph::Edge_Buffer_type Edge_Buffer_t>
  Directed_Vertex_Degree_Op(const Vertex_Buffer_From_t&, const Vertex_Buffer_To_t&,
                            const Edge_Buffer_t&)
      ->Directed_Vertex_Degree_Op<Vertex_Buffer_From_t, Vertex_Buffer_To_t, Edge_Buffer_t>;
  template <Sycl_Graph::Vertex_Buffer_type Vertex_Buffer_From_t,
            Sycl_Graph::Vertex_Buffer_type Vertex_Buffer_To_t,
            Sycl_Graph::Edge_Buffer_type Edge_Buffer_t>
  struct Undirected_Vertex_Degree_Op {
    typedef uint32_t Result_t;
    static constexpr Operation_Target_t operation_target = Operation_Target_Edge;
    static constexpr Operation_Type_t operation_type = Operation_Direct_Transform;
    static constexpr sycl::access_mode result_access_mode = sycl::access::mode::atomic;
    
    typedef typename Edge_Buffer_t::Edge_t Edge_t;
    typedef typename Vertex_Buffer_From_t::Vertex_t From_t;
    typedef typename Vertex_Buffer_To_t::Vertex_t To_t;

    Undirected_Vertex_Degree_Op() = default;
    Undirected_Vertex_Degree_Op(const Vertex_Buffer_From_t&, const Vertex_Buffer_To_t&,
                                const Edge_Buffer_t&) {}


    void operator()(const auto& edge_acc, const auto& from_acc, const auto& to_acc,
                    auto& result_acc, sycl::handler& h) {
      h.parallel_for(edge_acc.size(), [=](sycl::id<1> id) {
        auto& to_id = edge_acc[id].to;
        auto& from_id = edge_acc[id].from;
        if (to_id == Edge_t::invalid_id || from_id == Edge_t::invalid_id) return;
        result_acc[to_id]++;
        result_acc[from_id]++;
      });
    }
  };

  template <Sycl_Graph::Vertex_Buffer_type Vertex_Buffer_From_t,
            Sycl_Graph::Vertex_Buffer_type Vertex_Buffer_To_t,
            Sycl_Graph::Edge_Buffer_type Edge_Buffer_t>
  Undirected_Vertex_Degree_Op(const Vertex_Buffer_From_t&, const Vertex_Buffer_To_t&,
                              const Edge_Buffer_t&)
      -> Undirected_Vertex_Degree_Op<Vertex_Buffer_From_t, Vertex_Buffer_To_t, Edge_Buffer_t>;

}  // namespace Sycl_Graph::Sycl
#endif