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
#include <Sycl_Graph/Buffer/Sycl/Edge_Buffer.hpp>
#include <Sycl_Graph/Buffer/Sycl/Vertex_Buffer.hpp>
#include <Sycl_Graph/Graph/Base/Graph.hpp>
#include <type_traits>
namespace Sycl_Graph::Sycl
{

  template <Vertex_Buffer_type _Vertex_Buffer, Edge_Buffer_type _Edge_Buffer>
  struct Graph : public Sycl_Graph::Base::Graph<_Vertex_Buffer, _Edge_Buffer>

  {
    typedef Sycl_Graph::Base::Graph<_Vertex_Buffer, _Edge_Buffer> Base_t;
    typedef _Vertex_Buffer Vertex_Buffer;
    typedef _Edge_Buffer Edge_Buffer;
    typedef typename Base_t::uI_t uI_t;
    typedef typename Base_t::Vertex_t Vertex_t;
    typedef typename Base_t::Vertex_Data_t Vertex_Data_t;
    typedef typename Base_t::Edge_t Edge_t;
    typedef typename Base_t::Edge_Data_t Edge_Data_t;
    // create copy constructor
    Graph(sycl::queue &q, uI_t NV = 0, uI_t NE = 0, const sycl::property_list &props = {})
        : q(q), Base_t(Vertex_Buffer(q, NV, props), Edge_Buffer(q, NE, props)) {}

    Graph(sycl::queue &q, const std::vector<Vertex_t> &vertices,
          const std::vector<Edge_t> &edges = {},
          const sycl::property_list &props = {})
        : q(q), Base_t(Vertex_Buffer(q, vertices, props), Edge_Buffer(q, edges, props)) {}
    sycl::queue &q;
    uI_t& Graph_ID = this->Graph_ID;
    uI_t N_vertices() const
    {
      return this->vertex_buf.size();
    }
    uI_t N_edges() const
    {
      return this->edge_buf.current_size();
    }

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


  // // find vertex index based on condition
  // template <typename T> uI_t find(T condition) {
  //   uI_t idx = Vertex_t::invalid_id;
  //   sycl::buffer<uI_t, 1> res_buf(&idx, 1);
  //   q.submit([&](sycl::handler &h) {
  //     auto out = res_buf.template get_access<sycl::access::mode::write>(h);
  //     auto vertex_acc =
  //         vertex_buf.template get_access<sycl::access::mode::read>(h);
  //     find(out, vertex_acc, condition, h);
  //   });
  //   q.wait();
  //   return idx;
  // }

  // template <typename T0, typename T1, typename T2>
  // void find(T0 &res_acc, T1 &v_acc, T2 condition, sycl::handler &h) {
  //   h.parallel_for<class vertex_id_search>(sycl::range<1>(v_acc.size()),
  //                                          [=](sycl::id<1> id) {
  //                                            if (condition(v_acc[id[0]]))
  //                                              res_acc[0] = id[0];
  //                                          });
  // }

    // template <sycl::access::mode mode>
    // auto get_vertex_access(sycl::handler &h)
    // {
    //   return vertex_buf.template get_access<mode>(h);
    // }

    // template <sycl::access::mode mode>
    // auto get_edge_access(sycl::handler &h)
    // {
    //   return edge_buf.template get_access<mode>(h);
    // }
  };
} // namespace Sycl_Graph::Sycl

#endif
