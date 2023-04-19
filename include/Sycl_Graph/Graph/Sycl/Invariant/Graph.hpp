//
// Created by arch on 9/29/22.
//

#ifndef SYCL_GRAPH_SYCL_GRAPH_HPP
#define SYCL_GRAPH_SYCL_GRAPH_HPP
#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <iterator>
#include <limits>
#include <mutex>
#include <numeric>
#include <stddef.h>
#include <stdexcept>
#include <stdint.h>
#include <type_traits>
#include <utility>
// #include <Sycl_Graph/execution.hpp>
#include <CL/sycl.hpp>
#include <Sycl_Graph/Buffer/Sycl/Invariant/Edge_Buffer.hpp>
#include <Sycl_Graph/Buffer/Sycl/Invariant/Vertex_Buffer.hpp>
#include <Sycl_Graph/Graph/Invariant/Graph.hpp>
#include <type_traits>
#include <concepts>
namespace Sycl_Graph::Sycl::Invariant
{

  template <Sycl_Graph::Sycl::Invariant::Vertex_Buffer_type _Vertex_Buffer, Sycl_Graph::Sycl::Invariant::Edge_Buffer_type _Edge_Buffer>
  struct Graph : public Sycl_Graph::Invariant::Graph<_Vertex_Buffer, _Edge_Buffer>

  {
    typedef Sycl_Graph::Invariant::Graph<_Vertex_Buffer, _Edge_Buffer> Base_t;
    typedef _Vertex_Buffer Vertex_Buffer_t;
    typedef _Edge_Buffer Edge_Buffer_t;
    typedef typename Base_t::uI_t uI_t;
    typedef typename Base_t::Vertex_t Vertex_t;
    typedef typename Base_t::Vertex_Data_t Vertex_Data_t;
    typedef typename Base_t::Edge_t Edge_t;
    typedef typename Base_t::Edge_Data_t Edge_Data_t;
    Graph(sycl::queue &q, uI_t NV = 0, uI_t NE = 0, const sycl::property_list &props = {})
        : q(q), Base_t(Vertex_Buffer_t(q, NV, props), Edge_Buffer_t(q, NE, props)) {}

    Graph(Vertex_Buffer_t &vertex_buf, Edge_Buffer_t &edge_buf, sycl::queue &q)
        : q(q), Base_t(std::move(vertex_buf), std::move(edge_buf)) {}

    sycl::queue &q;

    uI_t max_vertices() const
    {
      return this->vertex_buf.max_size();
    }

    uI_t max_edges() const
    {
      return this->edge_buf.max_size();
    }

    void resize(uI_t NV_new, uI_t NE_new)
    {
      this->vertex_buf.resize(NV_new);
      this->edge_buf.resize(NE_new);
    }



    template <sycl::access_mode Mode, typename T, typename D = void>
    auto get_access(sycl::handler &h)
    {
      static_assert(Vertex_Buffer_t::template is_Vertex_type<T> || Edge_Buffer_t::template is_Edge_type<T>, "Type is not a vertex or edge type");
      if constexpr (Vertex_Buffer_t::template is_Vertex_type<T>)
      {
        auto acc = this->vertex_buf.template get_access<Mode, T, D>(h);
        // static_assert(std::is_same_v<typename decltype(acc)::Vertex_t, T>, "Vertex type mismatch");
      }
      else
      {
        auto acc = this->edge_buf.template get_access<Mode, T, D>(h);
        // static_assert(std::is_same_v<typename decltype(acc)::Edge_t, T>, "Edge type mismatch");
      }
    }

  };

  template <typename T>
  concept Graph_type = true;
} // namespace Sycl_Graph::Sycl

#endif
