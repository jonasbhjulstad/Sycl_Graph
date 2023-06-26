#ifndef SYCL_GRAPH_ALGORITHMS_PROPERTIES_EDGE_OPERATIONS_HPP
#define SYCL_GRAPH_ALGORITHMS_PROPERTIES_EDGE_OPERATIONS_HPP
#include <Sycl_Graph/Algorithms/Operations/Operation_Types.hpp>
#include <Sycl_Graph/Graph/Sycl/Graph.hpp>
namespace Sycl_Graph::Sycl {
  template <Sycl_Graph::Edge_Buffer_type Edge_Buffer_t, typename Derived>
  struct Edge_Extract_Operation {
    typedef typename Edge_Buffer_t::Edge_t Edge_t;
    typedef typename Edge_t::From_t From_t;
    typedef typename Edge_t::To_t To_t;
    typedef std::tuple<Edge_t, From_t, To_t> Accessor_Types;
    static constexpr std::array<sycl::access_mode, 3> graph_access_modes
        = {sycl::access_mode::read, sycl::access_mode::read, sycl::access_mode::read};
    static constexpr sycl::access_mode target_access_mode = sycl::access_mode::write;
    void _invoke(auto& accessors, auto& target_acc, sycl::handler& h) {
      auto& edge_acc = std::get<0>(accessors);
      auto& from_acc = std::get<1>(accessors);
      auto& to_acc = std::get<2>(accessors);
      static_cast<Derived*>(this)->invoke(edge_acc, from_acc, to_acc, target_acc, h);
    }

    template <Graph_type Graph_t> size_t target_buffer_size(const Graph_t& G) const {
      return static_cast<const Derived*>(this)->target_buffer_size(G);
    }
  };

  template <typename T>
  concept Edge_Extract_Operation_type = requires {
                                          typename T::Edge_Buffer_t;
                                          typename T::Edge_t;
                                          typename T::From_t;
                                          typename T::To_t;
                                          typename T::Accessor_Types;
                                          {
                                            T::graph_access_modes
                                            } -> std::same_as<std::array<sycl::access_mode, 3>>;
                                          {
                                            T::target_access_mode
                                            } -> std::same_as<sycl::access_mode>;
                                          { T::_invoke } -> std::same_as<void>;
                                        };

  template <Sycl_Graph::Edge_Buffer_type Edge_Buffer_t, typename Derived>
  struct Edge_Inject_Operation {
    typedef typename Edge_Buffer_t::Edge_t Edge_t;
    typedef typename Edge_t::From_t From_t;
    typedef typename Edge_t::To_t To_t;
    typedef std::tuple<Edge_t, From_t, To_t> Accessor_Types;
    static constexpr std::array<sycl::access_mode, 3> graph_access_modes
        = {sycl::access_mode::read_write, sycl::access_mode::read, sycl::access_mode::read};
    static constexpr sycl::access_mode target_access_mode = sycl::access_mode::read;
    void _invoke(auto& accessors, const auto& source_acc, sycl::handler& h) {
      auto& edge_acc = std::get<0>(accessors);
      auto& from_acc = std::get<1>(accessors);
      auto& to_acc = std::get<2>(accessors);
      static_cast<Derived*>(this)->invoke(edge_acc, from_acc, to_acc, source_acc, h);
    }

    template <Graph_type Graph_t> size_t source_buffer_size(const Graph_t& G) const {
      return static_cast<const Derived*>(this)->source_buffer_size(G);
    }
  };

  template <typename T>
  concept Edge_Inject_Operation_type = requires {
                                         typename T::Edge_Buffer_t;
                                         typename T::Edge_t;
                                         typename T::From_t;
                                         typename T::To_t;
                                         typename T::Accessor_Types;
                                           typename T::Source_t;
                                         {
                                           T::graph_access_modes
                                           } -> std::same_as<std::array<sycl::access_mode, 3>>;
                                         {
                                           T::target_access_mode
                                           } -> std::same_as<sycl::access_mode>;
                                         { T::_invoke } -> std::same_as<void>;
                                       };

  template <Sycl_Graph::Edge_Buffer_type Edge_Buffer_t, typename Derived>
  struct Edge_Transform_Operation {
    typedef typename Edge_Buffer_t::Edge_t Edge_t;
    typedef typename Edge_t::From_t From_t;
    typedef typename Edge_t::To_t To_t;
    typedef std::tuple<Edge_t, From_t, To_t> Accessor_Types;
    static constexpr std::array<sycl::access_mode, 3> graph_access_modes
        = {sycl::access_mode::read, sycl::access_mode::read, sycl::access_mode::read};
    static constexpr sycl::access_mode target_access_mode = sycl::access_mode::write;
    void _invoke(auto& accessors, auto& custom_acc, const auto& source_acc, auto& target_acc,
                 sycl::handler& h) {
      auto& edge_acc = std::get<0>(accessors);
      auto& from_acc = std::get<1>(accessors);
      auto& to_acc = std::get<2>(accessors);
      static_cast<Derived*>(this)->invoke(edge_acc, from_acc, to_acc, custom_acc, source_acc, target_acc,
                                                h);
    }

    template <Graph_type Graph_t> size_t source_buffer_size(const Graph_t& G) const {
      return static_cast<const Derived*>(this)->source_buffer_size(G);
    }

    template <Graph_type Graph_t> size_t target_buffer_size(const Graph_t& G) const {
      return static_cast<const Derived*>(this)->target_buffer_size(G);
    }
  };

  template <typename T>
  concept Edge_Transform_Operation_type = requires {
                                       typename T::Edge_Buffer_t;
                                       typename T::Edge_t;
                                       typename T::From_t;
                                       typename T::To_t;
                                       typename T::Accessor_Types;
                                         typename T::Source_t;
                                         typename T::Target_t;
                                       {
                                         T::graph_access_modes
                                         } -> std::same_as<std::array<sycl::access_mode, 3>>;
                                       { T::target_access_mode } -> std::same_as<sycl::access_mode>;
                                       { T::_invoke } -> std::same_as<void>;
                                     };

}  // namespace Sycl_Graph::Sycl

#endif
