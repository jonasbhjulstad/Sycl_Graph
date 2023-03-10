#ifndef SYCL_GRAPH_GRAPH_HPP
#define SYCL_GRAPH_GRAPH_HPP
#include <concepts>
#include <Sycl_Graph/Buffer/Base/Edge_Buffer.hpp>
#include <Sycl_Graph/Buffer/Base/Vertex_Buffer.hpp>
namespace Sycl_Graph::Base
{


  template <Vertex_Buffer_type _Vertex_Buffer_t, 
            Edge_Buffer_type _Edge_Buffer_t>
  struct Graph
  {


    Graph() = default;
    Graph(const _Vertex_Buffer_t& vertex_buffer, const _Edge_Buffer_t& edge_buffer)
        : vertex_buf(vertex_buffer), edge_buf(edge_buffer)
    {
    }
    Graph(const _Vertex_Buffer_t&& vertex_buffer, const _Edge_Buffer_t && edge_buffer)
        : vertex_buf(vertex_buffer), edge_buf(edge_buffer)
    {
    }
    typedef _Vertex_Buffer_t Vertex_Buffer_t;
    typedef _Edge_Buffer_t Edge_Buffer_t;

    typedef typename Vertex_Buffer_t::uI_t uI_t;
    typedef typename Vertex_Buffer_t::Vertex_t Vertex_t;
    typedef typename Vertex_Buffer_t::Data_t Vertex_Data_t;
    typedef typename Edge_Buffer_t::Edge_t Edge_t;
    typedef typename Edge_Buffer_t::Data_t Edge_Data_t;

    typedef Graph<Vertex_Buffer_t, Edge_Buffer_t> Graph_t;
    static constexpr auto invalid_id = Vertex_t::invalid_id;
    Vertex_Buffer_t vertex_buf;
    Edge_Buffer_t edge_buf;

    uI_t Graph_ID = 0;

    uI_t N_vertices() const { return vertex_buf.current_size(); }
    uI_t N_edges() const { return edge_buf.current_size(); }
    uI_t N_vertices_max() const { return vertex_buf.max_size(); }
    uI_t N_edges_max() const { return edge_buf.max_size(); }
    void resize(uI_t NV_new, uI_t NE_new)
    {
      vertex_buf.resize(NV_new);
      edge_buf.resize(NE_new);
    }

    auto& operator+(const Graph_t &other) const
    {
      this->vertex_buf + other.vertex_buf;
      this->edge_buf + other.edge_buf;
      return *this;
    }

    Graph_t &operator=(Graph_t &other)
    {
      vertex_buf = other.vertex_buf;
      edge_buf = other.edge_buf;
      return *this;
    }


    void add_vertex(const auto&& ... args)
    {
      vertex_buf.add(std::forward<decltype(args)>(args) ...);
    }

    void add_edge(const auto&& ... args)
    {
      edge_buf.add(std::forward<decltype(args)>(args) ...);
    }

    void remove_vertex(const auto&& ... args)
    {
      vertex_buf.remove(std::forward<decltype(args)>(args) ...);
    }

    void remove_edge(const auto&& ... args)
    {
      edge_buf.remove(std::forward<decltype(args)>(args) ...);
    }
    template <typename T>
    auto get_edges(const std::vector<uI_t>&& ids)
    {
      return edge_buf.template get_edges<T>(std::forward<decltype(ids)>(ids));
    }
    template <typename T>
    auto get_edges()
    {
      return edge_buf.template get_edges<T>();
    }
    auto get_edges()
    {
      return edge_buf.get_edges();
    }

    // // file I/O
    // void write_edgelist(std::string filename, std::string delimiter = ",",
    //                     bool edges_only = true)
    // {
    //   auto edges = edge_buf.get_edges();
    //   std::ofstream file(filename);
    //   file << "to" << delimiter << "from";
    //   if (!edges_only)
    //   {
    //     file << delimiter << "data";
    //   }
    //   file << "\n";

    //   write_edgelist(file, delimiter, edges_only);
    //   file.close();
    // }

    // void write_edgelist(std::ofstream &file, std::string delimiter = ",",
    //                     bool edges_only = true)
    // {
    //   auto edges = edge_buf.get_edges();
    //   for (auto e : edges)
    //   {
    //     file << delimiter << e.to << delimiter << e.from;
    //     if (!edges_only)
    //     {
    //       file << delimiter << e.data;
    //     }
    //     file << "\n";
    //   }
    // }

    // void write_vertexlist(std::string filename, std::string delimiter = ",")
    // {
    //   auto vertices = vertex_buf.get_vertices();
    //   std::ofstream file(filename);
    //   file << delimiter << "id" << delimiter << "data"
    //        << "\n";
    //   write_vertexlist(file, delimiter);
    //   file.close();
    // }

    // void write_vertexlist(std::ofstream &file, std::string delimiter = ",")
    // {
    //   auto vertices = vertex_buf.get_vertices();
    //   for (auto v : vertices)
    //   {
    //     file << delimiter << v.id << delimiter << v.data << "\n";
    //   }
    // }
  };

  template <typename T>
  concept Graph_type = requires(T t)
  {
    typename T::Vertex_Buffer_t;
    typename T::Edge_Buffer_t;
    typename T::uI_t;
    typename T::Vertex_t;
    typename T::Vertex_Data_t;
    typename T::Edge_t;
    typename T::Edge_Data_t;
    typename T::Graph_t;
    {t.vertex_buf};
    {t.edge_buf};
    {t.N_vertices()};
    {t.N_edges()};
    {t.resize(0, 0)};
    {t + t};
    {t = t};
    {t.add_vertex(0)};
    {t.add_edge(0, 0)};
    {t.remove_vertex(0)};
    {t.remove_edge(0, 0)};
    {t.template get_edges<int>()};
    {t.template get_edges<int>(std::vector<typename T::uI_t>{})};
    // {t.write_edgelist("")};
    // {t.write_edgelist(std::ofstream{})};
    // {t.write_vertexlist("")};
    // {t.write_vertexlist(std::ofstream{})};
  };


} // namespace Sycl_Graph
#endif