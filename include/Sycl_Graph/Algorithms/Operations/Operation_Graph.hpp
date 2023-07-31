#ifndef SYCL_GRAPH_ALGORITHMS_OPERATIONS_OPERATION_GRAPH_HPP
#define SYCL_GRAPH_ALGORITHMS_OPERATIONS_OPERATION_GRAPH_HPP
#include <Sycl_Graph/Algorithms/Operations/Operation_Types.hpp>
#include <Sycl_Graph/Buffer/Base/Buffer.hpp>
namespace Sycl_Graph::Sycl {
  struct Operation_Node : public Vertex<std::vector<std::shared_ptr<void>>> {
    Operation_Node() = default;
    template <typename... Ds> Operation_Node(const std::vector<Ds>&... data)
        : Base_t{std::vector<std::shared_ptr<void>>{std::make_shared<Ds>(data)...}} {}
  };

  typedef Buffer<Operation_Node> Operation_Node_Buffer;
  struct Operation_Link_Buffer {
    struct Operation_Link {
      virtual void initialize() = 0;
      virtual void invoke() = 0;
    };

    template <typename T> struct Operation_Link_Wrapper : public Operation_Link {
      const T& op;
      Operation_Link_Wrapper(const T& op) : op(op) {}
      void initialize() { op.initialize(); }
      void invoke() { op.invoke(); }
    };
    std::vector<std::unique_ptr<Operation_Link>> links;
    template <typename T> void add_link(const T& op) {
      links.emplace_back(std::make_unique<Operation_Link_Wrapper<T>>(op));
    }
  }

  template <typename Derived, Vertex_type _From_t, Vertex_type _To_t, typename... Custom_Ts>
  struct Operation_Edge
      : public Edge<std::tuple<std::shared_ptr<sycl::buffer<Custom_Ts>>...>, _From_t, _To_t>,
        public Operation_Link {
    using Base_t = Edge<std::tuple<std::shared_ptr<sycl::buffer<Custom_Ts>>, _From_t, _To_t>>;
    Operation_Edge(const Op& op) : op(op) {}
    void initialize() { static_cast<Derived*>(this)->initialize(); }
    void invoke() { static_cast<Derived*>(this)->invoke(); }
  };

}  // namespace Sycl_Graph::Sycl

#endif  // SYCL_GRAPH_ALGORITHMS_OPERATIONS_OPERATION_GRAPH_HPP
