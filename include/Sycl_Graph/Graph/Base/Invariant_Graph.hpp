#ifndef SYCL_GRAPH_INVARIANT_GRAPH_HPP
#define SYCL_GRAPH_INVARIANT_GRAPH_HPP
#include <CL/sycl.hpp>
#include <Sycl_Graph/Graph/Base/Graph.hpp>
#include <Sycl_Graph/Buffer/Base/Edge_Buffer_Pack.hpp>
#include <Sycl_Graph/Buffer/Base/Vertex_Buffer_Pack.hpp>
namespace Sycl_Graph
{


  template <Vertex_Buffer_Pack_type Vertex_Buffer_Pack_t, 
            Edge_Buffer_Pack_type _Edge_Buffer_Pack_t>
  struct Graph: public Sycl_Graph::Graph<Vertex_Buffer_Pack_t, _Edge_Buffer_Pack_t>
  {
    typedef Sycl_Graph::Graph<Vertex_Buffer_Pack_t, _Edge_Buffer_Pack_t> Base_t;
    typedef Vertex_Buffer_Pack_t Vertex_Buffer_t;
    typedef _Edge_Buffer_Pack_t Edge_Buffer_Pack_t;

    typedef typename Vertex_Buffer_t::uI_t uI_t;
    typedef typename Vertex_Buffer_t::Vertex_t Vertex_t;
    typedef typename Vertex_Buffer_t::Data_t Vertex_Data_t;
    typedef typename Edge_Buffer_Pack_t::Edge_t Edge_t;
    typedef typename Edge_Buffer_Pack_t::Data_t Edge_Data_t;
    Graph() = default;
    Graph(const Vertex_Buffer_t &vertex_buffer, const Edge_Buffer_Pack_t &edge_buffer) : Base_t(vertex_buffer, edge_buffer) {}

    template <typename T>
    uI_t current_size() const
    {
      if constexpr (Vertex_Buffer_t::template is_Vertex_type<T>)
        return this->vertex_buf.template current_size<T>();
      if constexpr(Edge_Buffer_Pack_t::template is_Edge_type<T>)
        return this->edge_buf.template current_size<T>();
    }

    template <typename T>
    uI_t max_size() const
    {
      if constexpr (Vertex_Buffer_t::template is_Vertex_type<T>)
        return this->vertex_buf.template max_size<T>();
      if constexpr(Edge_Buffer_Pack_t::template is_Edge_type<T>)
        return this->edge_buf.template max_size<T>();
    }

    uI_t max_size() const
    {
      return std::min(this->vertex_buf.max_size(), this->edge_buf.max_size());
    }

    template <typename T>
    void resize(const uI_t &new_size)
    {
      if constexpr (Vertex_Buffer_t::template is_Vertex_type<T>)
        this->vertex_buf.template resize<T>(new_size);
      if constexpr(Edge_Buffer_Pack_t::template is_Edge_type<T>)
        this->edge_buf.template resize<T>(new_size);
    }

    static constexpr auto invalid_id = std::numeric_limits<uI_t>::max();

  };

  template <typename T>
  concept Graph_type = 
  Sycl_Graph::Vertex_Buffer_Pack_type<typename T::Vertex_Buffer_t> &&
  Sycl_Graph::Edge_Buffer_Pack_type<typename T::Edge_Buffer_Pack_t>;


  

} // namespace Sycl_Graph
#endif