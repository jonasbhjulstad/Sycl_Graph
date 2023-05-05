#ifndef SYCL_GRAPH_BUFFER_SYCL_VERTEX_BUFFER_HPP
#define SYCL_GRAPH_BUFFER_SYCL_VERTEX_BUFFER_HPP
#include <CL/sycl.hpp>
#include <type_traits>
#include <concepts>
#include <Sycl_Graph/Buffer/Sycl/Buffer.hpp>
#include <Sycl_Graph/Buffer/Sycl/Buffer_Routines.hpp>
namespace Sycl_Graph::Sycl {

template <sycl::access::mode Mode, Sycl_Graph::Vertex_type Vertex_t>
struct Vertex_Accessor: public Buffer_Accessor<Mode, typename Vertex_t::ID_t, typename Vertex_t::Data_t>
{
  typedef typename Vertex_t::ID_t ID_t;
  typedef typename Vertex_t::Data_t Data_t;
  typedef Buffer_Accessor<Mode, typename Vertex_t::ID_t, typename Vertex_t::Data_t> Base_t;
  Vertex_Accessor(Base_t&& base): Base_t(base) {}
  

  sycl::accessor<ID_t, 1, Mode>  ids = std::get<0>(this->accessors);
  sycl::accessor<Data_t, 1, Mode> data = std::get<1>(this->accessors);
  Vertex_t operator[](ID_t idx) const
  {
    auto [id, data] = this->get_idx(idx);
    return Vertex_t(id, data);
  }
};


template <Sycl_Graph::Vertex_type _Vertex_t,
          std::unsigned_integral _uI_t = uint32_t>
struct Vertex_Buffer : public Buffer<_uI_t, typename _Vertex_t::ID_t,
                                     typename _Vertex_t::Data_t> {

  typedef Buffer<_uI_t, typename _Vertex_t::ID_t,
                                     typename _Vertex_t::Data_t> Base_t;
  typedef _Vertex_t Vertex_t;
  typedef typename Base_t::Data_t Data_t;
  typedef typename Vertex_t::ID_t ID_t;
  typedef typename Vertex_t::Data_t Vertex_Data_t;
  typedef _uI_t uI_t;

  sycl::queue &q = Base_t::q;
  Vertex_Buffer(sycl::queue &q, uI_t NV = 1, const sycl::property_list &props = {})
      : Base_t(q, NV, props) {}

  Vertex_Buffer(sycl::queue &q, const std::vector<ID_t> &ids,
                const std::vector<Data_t> &data = {},
                const sycl::property_list &props = {})
      : Base_t(q, ids, data, props) {}

  Vertex_Buffer(sycl::queue &q, const std::vector<Vertex_t> &vertices,
                const sycl::property_list &props = {})
      : Base_t(q, 1, props) {
    this->add(vertices);
  }

  template <typename T>
  static constexpr bool has_Vertex_type()
  {
    return std::is_same_v<T, Vertex_t>;
  }


  std::vector<ID_t> get_valid_ids() {
      auto& id_buf = this->template get_buffer<ID_t>();
      std::vector<ID_t> ids = buffer_get(id_buf);
      ids.erase(std::remove_if(ids.begin(), ids.end(), [](const ID_t& id){return id == Vertex_t::invalid_id;}), ids.end());
      return ids;
  }

  void add(const std::vector<Vertex_t>& vertices)
  {
    std::vector<ID_t> ids;
    std::vector<Vertex_Data_t> data;
    data.reserve(vertices.size());
    ids.reserve(vertices.size());
    for (const auto &v : vertices) {
      ids.push_back(v.id);
      data.push_back(v.data);
    }
    auto tup = std::make_tuple(ids, data);
    static_cast<Base_t*>(this)->add(tup);
  }

  uI_t N_vertices() const { return this->current_size(); }

  std::vector<Vertex_t> get_vertices() {
    auto Vertex_tuple = this->template get<ID_t, Data_t>();
    std::vector<Vertex_t> vertices;
    vertices.reserve(Vertex_tuple.first.size());
    for (size_t i = 0; i < Vertex_tuple.first.size(); i++) {
      vertices.push_back(
          Vertex_t(Vertex_tuple.first[i], Vertex_tuple.second[i]));
    }
    return vertices;
  }

  template <sycl::access_mode Mode>
  Vertex_Accessor<Mode, Vertex_t> get_access(sycl::handler& h)
  {
    return Vertex_Accessor<Mode, Vertex_t>(static_cast<Base_t*>(this)->template get_access<Mode, typename Vertex_t::ID_t, typename Vertex_t::Data_t>(h));
  }

  void remove(const std::vector<ID_t> &ids) {
    this->template remove_elements<ID_t>(ids);
  }
};

template <typename T>
concept Vertex_Buffer_type = Sycl_Graph::Vertex_Buffer_type<T>;

} // namespace Sycl_Graph::Sycl

#endif // SYCL_GRAPH_SYCL_VERTEX_BUFFER_HPP