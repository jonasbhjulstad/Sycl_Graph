#ifndef SYCL_GRAPH_SYCL_EDGE_BUFFER_HPP
#define SYCL_GRAPH_SYCL_EDGE_BUFFER_HPP
#include <CL/sycl.hpp>
#include <Sycl_Graph/Graph/Base/Graph_Types.hpp>
#include <Sycl_Graph/Buffer/Sycl/Buffer.hpp>
#include <Sycl_Graph/Buffer/Base/Edge_Buffer.hpp>
#include <Sycl_Graph/Buffer/Sycl/Buffer_Routines.hpp>
namespace Sycl_Graph::Sycl {
template <sycl::access::mode Mode, Sycl_Graph::Edge_type Edge_t>
struct Edge_Accessor: public Buffer_Accessor<Mode, typename Edge_t::Connection_IDs, typename Edge_t::Data_t>
{
  typedef typename Edge_t::Connection_IDs Connection_IDs;
  typedef typename Edge_t::Data_t Data_t;
  typedef Buffer_Accessor<Mode, Connection_IDs, Data_t> Base_t;
  Edge_Accessor(const Base_t& base): Base_t(base){}

  sycl::accessor<Connection_IDs, 1, Mode>  ids = std::get<0>(this->accessors);
  sycl::accessor<Data_t, 1, Mode> data = std::get<1>(this->accessors);
  Edge_t operator[](sycl::id<1> idx) const
  {
    const auto& id = this->ids[idx];
    const auto& data = this->data[idx];
    return Edge_t(id, data);
  }
};

template <Sycl_Graph::Edge_type _Edge_t, std::unsigned_integral _uI_t = uint32_t> 
struct Edge_Buffer: public Buffer<_uI_t, typename _Edge_t::Connection_IDs, typename _Edge_t::Data_t>
 {
  typedef _Edge_t Edge_t;
  typedef Buffer<_uI_t, typename _Edge_t::Connection_IDs, typename _Edge_t::Data_t> Base_t;
  typedef typename Base_t::Data_t Data_t;
  typedef typename Edge_t::ID_t ID_t;
  typedef typename Edge_t::Connection_IDs Connection_IDs;
  typedef _uI_t uI_t;

  

  sycl::queue& q = Base_t::q;
  Edge_Buffer(sycl::queue &q, uI_t NE = 1, const sycl::property_list &props = {}): Base_t(q, NE, props){}

  Edge_Buffer(sycl::queue &q, const std::vector<Connection_IDs>& ids, const std::vector<Data_t>& data = {}, const sycl::property_list &props = {}): Base_t(q, ids, data, props){}

  Edge_Buffer(sycl::queue &q, const std::vector<Edge_t> &edges,
              const sycl::property_list &props = {}): Base_t(q, 0, props)
              {
                this->add(edges);
              }


  std::vector<Connection_IDs> get_valid_ids() {
      auto& id_buf = this->template get_buffer<Connection_IDs>();
      std::vector<Connection_IDs> ids = buffer_get(id_buf);
      ids.erase(std::remove_if(ids.begin(), ids.end(), [](const Connection_IDs& id){return id.to == Connection_IDs::invalid_id || id.from == Connection_IDs::invalid_id;}), ids.end());
      return ids;
  }

  uI_t N_edges() const { return this->current_size(); }

  void add(const std::vector<Edge_t>& edges)
  {
      std::vector<Connection_IDs> ids;
      std::vector<typename Edge_t::Data_t> data;
      data.reserve(edges.size());
      ids.reserve(edges.size());
      for (const auto& e: edges)
      {
          ids.push_back(e.ids);
          data.push_back(e.data);
      }
      static_cast<Base_t*>(this)->add(std::make_tuple(ids, data));
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

  template <sycl::access_mode Mode>
  auto get_access(sycl::handler& h)
  {
    return Edge_Accessor<Mode, Edge_t>(std::move(static_cast<Base_t*>(this)->template get_access<Mode, Connection_IDs, typename Edge_t::Data_t>(h)));
  }

};

template <typename T>
concept Edge_Buffer_type = Sycl_Graph::Edge_Buffer_type<T>;
} // namespace sycl_graph

#endif // 