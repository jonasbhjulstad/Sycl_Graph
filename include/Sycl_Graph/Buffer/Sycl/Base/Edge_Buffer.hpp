#ifndef SYCL_GRAPH_SYCL_EDGE_BUFFER_HPP
#define SYCL_GRAPH_SYCL_EDGE_BUFFER_HPP
#include <CL/sycl.hpp>
#include <Sycl_Graph/Graph/Base/Graph_Types.hpp>
#include <Sycl_Graph/Buffer/Sycl/Base/Buffer.hpp>
#include <Sycl_Graph/Buffer/Base/Edge_Buffer.hpp>
#include <Sycl_Graph/Buffer/Sycl/Buffer_Routines.hpp>
namespace Sycl_Graph::Sycl::Base {
template <Sycl_Graph::Base::Edge_type Edge_t, sycl::access::mode Mode>
struct Edge_Accessor {
  typedef typename Edge_t::Data_t Data_t;
  typedef typename Edge_t::Connection_IDs Connection_IDs;
  Edge_Accessor(sycl::buffer<typename Edge_t::Connection_IDs, 1> &id_buf, sycl::buffer<typename Edge_t::Data_t, 1>& data_buf, sycl::handler &h,
                sycl::property_list props = {})
      : ids(id_buf, h, props), data(data_buf, h, props) {}

  Edge_t operator[](std::size_t i){
    return Edge_t(ids[i], data[i]);}
  sycl::accessor<Connection_IDs, 1, Mode> ids;
  sycl::accessor<Data_t, 1, Mode> data;
  std::size_t size() const { return ids.size(); }
};

template <Sycl_Graph::Base::Edge_type _Edge_t, std::unsigned_integral _uI_t = uint32_t> 
struct Edge_Buffer: public Buffer<_uI_t, typename _Edge_t::Connection_IDs, typename _Edge_t::Data_t>
 {
  typedef _Edge_t Edge_t;
  typedef typename Edge_t::ID_t ID_t;
  typedef Edge_t Data_t;
  typedef typename Edge_t::Data_t Edge_Data_t;
  typedef typename Edge_t::Connection_IDs Connection_IDs;
  typedef _uI_t uI_t;
  typedef Buffer<_uI_t, Connection_IDs, Edge_Data_t> Base_t;

  sycl::queue& q = Base_t::q;
  Edge_Buffer(sycl::queue &q, uI_t NE = 1, const sycl::property_list &props = {}): Base_t(q, NE, props){}

  Edge_Buffer(sycl::queue &q, const std::vector<Connection_IDs>& ids, const std::vector<Data_t>& data = {}, const sycl::property_list &props = {}): Base_t(q, ids, data, props){}

  Edge_Buffer(sycl::queue &q, const std::vector<Edge_t> &edges,
              const sycl::property_list &props = {}): Base_t(q, 0, props)
              {
                this->add(edges);
              }
              

  std::vector<Connection_IDs> get_valid_ids() {
      return this->template get<Connection_IDs>([](const auto& e){return e.is_valid();});
  }

  uI_t N_edges() const { return this->current_size(); }

  void add(const std::vector<Edge_t>& edges)
  {
      std::vector<Connection_IDs> ids;
      std::vector<Edge_Data_t> data;
      data.reserve(edges.size());
      ids.reserve(edges.size());
      for (const auto& e: edges)
      {
          ids.push_back(e.ids);
          data.push_back(e.data);
      }
      this->template assign_add<Connection_IDs>(ids, data);
  }

  template <sycl::access_mode Mode>
  Edge_Accessor<Edge_t, Mode> get_access(sycl::handler& h)
  {
    auto [id_buf, data_buf] = this->get_buffers();
    return Edge_Accessor<Edge_t, Mode>(id_buf, data_buf, h);
  }

  std::vector<Edge_t> get_edges()
  {
    auto edge_tuple = this->template get<Connection_IDs, Data_t>();
    std::vector<Edge_t> edges;
    edges.reserve(edge_tuple.first.size());
    for (size_t i = 0; i < edge_tuple.first.size(); i++)
    {
        edges.push_back(Edge_t(edge_tuple.first[i], edge_tuple.second[i]));
    }
    return edges;
  }

  void remove(const std::vector<Connection_IDs>& ids)
  {
    this->template remove_elements<Connection_IDs>(ids);
  }

};

template <typename T>
concept Edge_Buffer_type = Sycl_Graph::Base::Edge_Buffer_type<T>;
} // namespace sycl_graph

#endif // 