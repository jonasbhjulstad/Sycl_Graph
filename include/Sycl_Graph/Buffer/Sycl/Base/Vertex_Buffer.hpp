#ifndef SYCL_GRAPH_BUFFER_SYCL_VERTEX_BUFFER_HPP
#define SYCL_GRAPH_BUFFER_SYCL_VERTEX_BUFFER_HPP
#include <CL/sycl.hpp>
#include <type_traits>
#include <concepts>
#include <Sycl_Graph/Buffer/Sycl/Base/Buffer.hpp>
#include <Sycl_Graph/Buffer/Sycl/Buffer_Routines.hpp>
namespace Sycl_Graph::Sycl::Base {

template <Sycl_Graph::Base::Vertex_type Vertex_t, sycl::access::mode Mode>
struct Vertex_Accessor {
  typedef typename Vertex_t::Data_t Data_t;
  typedef typename Vertex_t::ID_t ID_t;
  Vertex_Accessor(sycl::buffer<ID_t, 1> &id_buf, sycl::buffer<Data_t, 1>& data_buf, sycl::handler &h,
                sycl::property_list props = {})
      : id(id_buf, h, props), data(data_buf, h, props) {}

  Vertex_t operator[](std::size_t i) const{
    return Vertex_t(id[i], data[i]);}

  Vertex_t operator[](sycl::id<1> i) const
  {
    return Vertex_t(id[i], data[i]);}

  sycl::accessor<ID_t, 1, Mode> id;
  sycl::accessor<Data_t, 1, Mode> data;
};


template <Sycl_Graph::Base::Vertex_type _Vertex_t,
          std::unsigned_integral _uI_t = uint32_t>
struct Vertex_Buffer : public Buffer<_uI_t, typename _Vertex_t::ID_t,
                                     typename _Vertex_t::Data_t> {

  typedef _Vertex_t Vertex_t;
  typedef typename Vertex_t::ID_t ID_t;
  typedef Vertex_t Data_t;
  typedef typename Vertex_t::Data_t Vertex_Data_t;
  typedef _uI_t uI_t;
  typedef Buffer<_uI_t, typename _Vertex_t::ID_t,
                                     typename _Vertex_t::Data_t> Base_t;

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

  std::vector<ID_t> get_valid_ids() {
    return this->template get<ID_t>(
        [](const auto &v) { return v.is_valid(); });
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
    this->template assign_add<ID_t>(ids, data);
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
  Vertex_Accessor<Vertex_t, Mode> get_access(sycl::handler& h)
  {
    auto [id_buf, data_buf] = this->get_buffers();
    return Vertex_Accessor<Vertex_t, Mode>(id_buf, data_buf, h);
  }

  void remove(const std::vector<ID_t> &ids) {
    this->template remove_elements<ID_t>(ids);
  }
};

template <typename T>
concept Vertex_Buffer_type = Sycl_Graph::Base::Vertex_Buffer_type<T>;

} // namespace Sycl_Graph::Sycl

#endif // SYCL_GRAPH_SYCL_VERTEX_BUFFER_HPP