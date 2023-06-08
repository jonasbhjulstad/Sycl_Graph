export module Operations.Vertex;
#include <Sycl_Graph/Common/common.hpp>
template <Vertex_Buffer_type Vertex_Buffer_t, typename Derived>
export struct Vertex_Extract_Operation {
  typedef typename Vertex_Buffer_t::Vertex_t Vertex_t;
  typedef std::tuple<Vertex_t> Accessor_Types;
  static constexpr std::array<sycl::access_mode, 1> graph_access_modes = {sycl::access_mode::read};
  static constexpr sycl::access_mode target_access_mode = sycl::access_mode::write;
  void _invoke(auto& accessors, auto& target_acc, sycl::handler& h) {
    auto& v_acc = std::get<0>(accessors);
    static_cast<const Derived*>(this)->invoke(v_acc, target_acc, h);
  }

  template <Graph_type Graph_t> size_t target_buffer_size(const Graph_t& G) const {
    return static_cast<const Derived*>(this)->target_buffer_size(G);
  }
};

export template <typename T> concept Vertex_Extract_Operation_type
    = requires {
        typename T::Vertex_Buffer_t;
        typename T::Vertex_t;
        typename T::Accessor_Types;
        typename T::Target_t;
        { T::graph_access_modes } -> std::same_as<std::array<sycl::access_mode, 1>>;
        { T::target_access_mode } -> std::same_as<sycl::access_mode>;
        { T::_invoke } -> std::same_as<void>;
      };

template <Vertex_Buffer_type Vertex_Buffer_t, typename Derived>
export struct Vertex_Inject_Operation {
  typedef typename Vertex_Buffer_t::Vertex_t Vertex_t;
  typedef std::tuple<Vertex_t> Accessor_Types;
  static constexpr std::array<sycl::access_mode, 1> graph_access_modes
      = {sycl::access_mode::read_write};
  void _invoke(auto& accessors, const auto& source_acc, sycl::handler& h) {
    auto& v_acc = std::get<0>(accessors);
    static_cast<const Derived*>(this)->invoke(v_acc, source_acc, h);
  }

  template <Graph_type Graph_t> size_t source_buffer_size(const Graph_t& G) const {
    return static_cast<const Derived*>(this)->source_buffer_size(G);
  }
};

template <Vertex_Buffer_type Vertex_Buffer_t, typename Derived>
export struct Vertex_Direct_Inject_Op : public Vertex_Inject_Operation<Vertex_Buffer_t, Derived> {
  void _invoke(auto& accessors, const auto& source_acc, sycl::handler& h) {
    auto& v_acc = std::get<0>(accessors);
    h.parallel_for(v_acc.size(), [&](sycl::id<1> id) { v_acc[id].data = source_acc[id]; });
  }
};

export template <typename T> concept Vertex_Inject_Operation_type
    = requires {
        typename T::Vertex_Buffer_t;
        typename T::Vertex_t;
        typename T::Accessor_Types;
        { T::graph_access_modes } -> std::same_as<std::array<sycl::access_mode, 1>>;
        { T::_invoke } -> std::same_as<void>;
      };

template <Vertex_Buffer_type Vertex_Buffer_t, typename Derived>
export struct Vertex_Transform_Operation {
  typedef typename Vertex_Buffer_t::Vertex_t Vertex_t;
  typedef std::tuple<Vertex_t> Accessor_Types;
  static constexpr std::array<sycl::access_mode, 1> graph_access_modes = {sycl::access_mode::read};
  void _invoke(auto& accessors, const auto& source_acc, auto& target_acc, sycl::handler& h) {
    auto& v_acc = std::get<0>(accessors);
    static_cast<const Derived*>(this)->invoke(v_acc, source_acc, target_acc, h);
  }

  template <Graph_type Graph_t> size_t source_buffer_size(const Graph_t& G) const {
    return static_cast<const Derived*>(this)->source_buffer_size(G);
  }

  template <Graph_type Graph_t> size_t target_buffer_size(const Graph_t& G) const {
    return static_cast<const Derived*>(this)->target_buffer_size(G);
  }
};

export template <typename T> concept Vertex_Transform_Operation_type
    = requires() {
        typename T::Vertex_Buffer_t;
        typename T::Vertex_t;
        typename T::Accessor_Types;
        { T::graph_access_modes } -> std::same_as<std::array<sycl::access_mode, 1>>;
        { T::_invoke } -> std::same_as<void>;
      };