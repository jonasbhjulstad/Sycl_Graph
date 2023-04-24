//
// Created by arch on 9/29/22.
//

#ifndef SYCL_GRAPH_SYCL_GRAPH_HPP
#define SYCL_GRAPH_SYCL_GRAPH_HPP
#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <iterator>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
// #include <Sycl_Graph/execution.hpp>
#include <CL/sycl.hpp>
#include <Sycl_Graph/Buffer/Sycl/Buffer_Pack.hpp>
#include <Sycl_Graph/Buffer/Sycl/Edge_Buffer.hpp>
#include <Sycl_Graph/Buffer/Sycl/Vertex_Buffer.hpp>
#include <Sycl_Graph/Graph/Base/Graph.hpp>
#include <Sycl_Graph/Graph/Base/Graph_Types.hpp>
#include <concepts>
#include <type_traits>
namespace Sycl_Graph::Sycl {

  template <Sycl_Graph::Sycl::Buffer_Pack_type _Vertex_Buffer_Pack,
            Sycl_Graph::Sycl::Buffer_Pack_type _Edge_Buffer_Pack>
  struct Graph : public Sycl_Graph::Graph<_Vertex_Buffer_Pack, _Edge_Buffer_Pack> {
    typedef Sycl_Graph::Graph<_Vertex_Buffer_Pack, _Edge_Buffer_Pack> Base_t;
    typedef _Vertex_Buffer_Pack Vertex_Buffer_Pack_t;
    typedef _Edge_Buffer_Pack Edge_Buffer_Pack_t;
    typedef typename Base_t::uI_t uI_t;
    Graph(sycl::queue &q, uI_t NV = 0, uI_t NE = 0, const sycl::property_list &props = {})
        : q(q), Base_t(Vertex_Buffer_t(q, NV, props), Edge_Buffer_t(q, NE, props)) {}

    Graph(Vertex_Buffer_t &vertex_buf, Edge_Buffer_t &edge_buf, sycl::queue &q)
        : q(q), Base_t(std::move(vertex_buf), std::move(edge_buf)) {}

    sycl::queue &q;

    uI_t max_vertices() const { return this->vertex_buf.max_size(); }

    uI_t max_edges() const { return this->edge_buf.max_size(); }

    void resize(uI_t NV_new, uI_t NE_new) {
      this->vertex_buf.resize(NV_new);
      this->edge_buf.resize(NE_new);
    }

    template <sycl::access_mode Mode, Vertex_type V> auto get_vertex_access(sycl::handler &h) {
      auto& buf = std::get<Vertex_Buffer<V>>(this->vertex_buf.buffers);
      return buf.template get_access<Mode>(h);
    }
    template <sycl::access_mode Mode, Edge_type E> auto get_edge_access(sycl::handler &h) {
      auto& buf = std::get<Edge_Buffer<E>>(this->edge_buf.buffers);
      return buf.template get_access<Mode>(h);
    }
  };

  template <typename T>
  concept Invariant_Graph_type = true;
}  // namespace Sycl_Graph::Sycl

#endif
