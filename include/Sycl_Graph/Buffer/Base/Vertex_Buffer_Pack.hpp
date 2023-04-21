#ifndef SYCL_GRAPH_INVARIANT_Vertex_Buffer_Pack_HPP
#define SYCL_GRAPH_INVARIANT_Vertex_Buffer_Pack_HPP
#include <Sycl_Graph/Buffer/Base/Vertex_Buffer.hpp>
#include <Sycl_Graph/Buffer/Base/Buffer_Pack.hpp>
#include <Sycl_Graph/type_helpers.hpp>
#include <concepts>

namespace Sycl_Graph {

  template <template <typename...> typename Base_Buffer_Pack_t,
            Sycl_Graph::Vertex_Buffer_type... VBs>
  struct Vertex_Buffer_Pack : public Base_Buffer_Pack_t<VBs...> {
    typedef typename std::tuple_element_t<0, std::tuple<VBs...>>::uI_t uI_t;

    typedef std::tuple<typename VBs::Vertex_t...> Vertex_t;
    typedef std::tuple<typename VBs::Data_t...> Data_t;
    typedef Base_Buffer_Pack_t<VBs...> Base_t;
    Vertex_Buffer_Pack() = default;
    Vertex_Buffer_Pack(const VBs &...buffers) : Base_t(buffers...) {}
    Vertex_Buffer_Pack(const VBs &&...buffers) : Base_t(buffers...) {}
    template <typename V> static constexpr bool is_Vertex_type
        = Sycl_Graph::is_Vertex_type<V> && has_type<V, Vertex_t>::value;

    static constexpr uI_t invalid_id = std::numeric_limits<uI_t>::max();
    template <typename V> using Vertex = Sycl_Graph::Vertex<V, uI_t>;

    // check if T is in ID_t
    template <typename T> static constexpr bool is_ID_type
        = std::disjunction_v<std::is_same<T, typename VBs::Vertex_t::ID_t>...>;

    template <typename T> static constexpr bool is_Data_type
        = std::disjunction_v<std::is_same<T, typename VBs::Vertex_t::Data_t>...>;

    template <typename V> void add(const std::vector<uI_t> &&ids, const std::vector<V> &&data) {
      std::vector<V> vertices(ids.size());
      vertices.reserve(ids.size());
      std::transform(ids.begin(), ids.end(), data.begin(), vertices.begin(),
                     [](auto &&id, auto &&data) {
                       return V{id, data};
                     });
      add(vertices);
    }

    template <typename D> void add(const std::vector<D> &&data) {
      std::vector<Vertex<D>> vertices(data.size());
      std::vector<uI_t> ids = get_buffer<D>().get_available_ids(data.size());
      vertices.reserve(data.size());
      for (uI_t i = 0; i < data.size(); ++i) vertices.emplace_back(ids[i], data[i]);
      get_buffer<Vertex<D>>().add(vertices);
    }

    template <typename... Ds> void add(const std::vector<Ds> &&...data) { (add(data), ...); }

    template <typename D> void add(const std::vector<uI_t> &&ids) {
      std::vector<Vertex<D>> vertices(ids.size());
      vertices.reserve(ids.size());
      for (uI_t i = 0; i < ids.size(); ++i) vertices.emplace_back(ids[i], D{});
      add(vertices);
    }

    template <typename V> void remove(const std::vector<uI_t> &&ids) {
      get_buffer<V>().remove(ids);
    }

    template <typename V> auto get_vertices() const { return get_buffer<V>().get_vertices(); }

    template <typename V> auto get_vertices(const std::vector<uI_t> &ids) const {
      return get_buffer<V>().get_vertices(ids);
    }

    auto get_vertices() const { return std::make_tuple((get_buffer<VBs>().get_vertices(), ...)); }
  };

  template <Sycl_Graph::Vertex_Buffer_type... VBs> struct Vertex_Buffer_Pack<Buffer_Pack, VBs...>;

  template <typename T>
  concept Vertex_Buffer_Pack_type = std::unsigned_integral<typename T::uI_t>;

}  // namespace Sycl_Graph
#endif