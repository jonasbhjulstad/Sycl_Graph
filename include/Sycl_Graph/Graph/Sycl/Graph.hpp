//
// Created by arch on 9/29/22.
//

#ifndef SYCL_GRAPH_SYCL_GRAPH_HPP
#define SYCL_GRAPH_SYCL_GRAPH_HPP
#include <algorithm>
#include <limits>
#include <numeric>
#include <CL/sycl.hpp>
#include <Sycl_Graph/Buffer/Base/Buffer_Pack.hpp>
#include <Sycl_Graph/Buffer/Sycl/Buffer.hpp>
#include <Sycl_Graph/Buffer/Sycl/Edge_Buffer.hpp>
#include <Sycl_Graph/Buffer/Sycl/Vertex_Buffer.hpp>
#include <Sycl_Graph/Graph/Base/Graph.hpp>
#include <Sycl_Graph/Graph/Base/Graph_Types.hpp>
#include <concepts>
#include <type_traits>
namespace Sycl_Graph::Sycl {

  template <Sycl_Graph::Sycl::Buffer_type _Vertex_Buffer,
            Sycl_Graph::Sycl::Buffer_type _Edge_Buffer>
  struct Graph : public Sycl_Graph::Graph<_Vertex_Buffer, _Edge_Buffer> {
    typedef Sycl_Graph::Graph<_Vertex_Buffer, _Edge_Buffer> Base_t;
    typedef _Vertex_Buffer Vertex_Buffer_t;
    typedef _Edge_Buffer Edge_Buffer_t;
    Graph(sycl::queue &q, uint32_t NV = 0, uint32_t NE = 0, const sycl::property_list &props = {})
        : q(q), Base_t(Vertex_Buffer_t(q, NV, props), Edge_Buffer_t(q, NE, props)) {}

    Graph(Vertex_Buffer_t &vertex_buf, Edge_Buffer_t &edge_buf, sycl::queue &q)
        : q(q), Base_t(std::move(vertex_buf), std::move(edge_buf)) {}

    sycl::queue &q;

    uint32_t max_vertices() const { return this->vertex_buf.max_size(); }

    uint32_t max_edges() const { return this->edge_buf.max_size(); }

    void resize(uint32_t NV_new, uint32_t NE_new) {
      this->vertex_buf.resize(NV_new);
      this->edge_buf.resize(NE_new);
    }

    template <Vertex_type V, sycl::access_mode Mode>
    using Vertex_Pack_Accessor_t = typename std::enable_if<Buffer_Pack_type<Vertex_Buffer_t>, Vertex_Accessor<Mode, V>>::type;

    template <Edge_type E, sycl::access_mode Mode>
    using Edge_Pack_Accessor_t = typename std::enable_if<Buffer_Pack_type<Edge_Buffer_t>, Edge_Accessor<Mode, E>>::type;

    template <sycl::access_mode Mode, Vertex_type V>
    Vertex_Pack_Accessor_t<V, Mode> get_vertex_access(sycl::handler &h) {
      auto &buf = std::get<Vertex_Buffer<V>>(this->vertex_buf.buffers);
      return buf.template get_access<Mode>(h);
    }
    template <sycl::access_mode Mode, Edge_type E>
    Edge_Pack_Accessor_t<E, Mode> get_edge_access(sycl::handler &h) {
      auto &buf = std::get<Edge_Buffer<E>>(this->edge_buf.buffers);
      return buf.template get_access<Mode>(h);
    }

    template <Vertex_type V, sycl::access_mode Mode>
    using Vertex_Accessor_t = typename std::enable_if<!Buffer_Pack_type<Vertex_Buffer_t>, Vertex_Accessor<Mode, V>>::type;
    template <Edge_type E, sycl::access_mode Mode>
    using Edge_Accessor_t = typename std::enable_if<!Buffer_Pack_type<Edge_Buffer_t>, Edge_Accessor<Mode, E>>::type;

    template <sycl::access_mode Mode, Vertex_type V>
    Vertex_Accessor_t<V, Mode> get_vertex_access(sycl::handler &h) {
      auto &buf = this->vertex_buf;
      return buf.template get_access<Mode>(h);
    }

    template <sycl::access_mode Mode, Edge_type E>
    Edge_Accessor_t<E, Mode> get_edge_access(sycl::handler &h) {
      auto &buf = this->edge_buf;
      return buf.template get_access<Mode>(h);
    }

    template <typename T>
    auto get_buffer()
    {
      if constexpr (is_Vertex_type<T>)
      {
        return this->vertex_buf.template get_buffer<typename T::ID_t, typename T::Data_t>();
      }
      else if constexpr (is_Edge_type<T>)
      {
        return this->edge_buf.template get_buffer<typename T::ID_Pair_t, typename T::Data_t>();
      }
    }

    template <sycl::access_mode Mode, typename T>
    auto get_access(sycl::handler& h)
    {
      if constexpr(is_Vertex_type<T>)
      {
        return get_vertex_access<Mode, T>(h);
      }
      else if constexpr(is_Edge_type<T>)
      {
        return get_edge_access<Mode, T>(h);
      }
    }


    template <typename T>
    size_t current_size() const
    {
      if constexpr(is_Edge_type<T>)
      {
        return this->edge_buf.template get_buffer<typename T::ID_Pair_t, typename T::Data_t>().current_size();
      }
      else if constexpr (is_Vertex_type<T>)
      {
        return this->vertex_buf.template get_buffer<typename T::ID_t, typename T::Data_t>().current_size();
      }

      return -1;
    }

  };

  template <typename T>
  concept Graph_type = true;
}  // namespace Sycl_Graph::Sycl

#endif
