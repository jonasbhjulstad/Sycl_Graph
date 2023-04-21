#ifndef SYCL_GRAPH_INVARIANT_EDGE_BUFFER_HPP
#define SYCL_GRAPH_INVARIANT_EDGE_BUFFER_HPP
#include <Sycl_Graph/Buffer/Base/Edge_Buffer.hpp>
#include <Sycl_Graph/Buffer/Base/Buffer_Pack.hpp>
#include <Sycl_Graph/type_helpers.hpp>

namespace Sycl_Graph {
  template <template <typename ...> typename Base_Buffer_Pack_t, Sycl_Graph::Edge_Buffer_type... EBs> struct Edge_Buffer_Pack
      : public Base_Buffer_Pack_t<EBs...> {
    typedef Base_Buffer_Pack_t<EBs...> Base_t;
    Edge_Buffer_Pack() = default;
    Edge_Buffer_Pack(const EBs &...buffers) : Base_t(buffers...) {}
    Edge_Buffer_Pack(const EBs &&...buffers) : Base_t(buffers...) {}

    typedef typename Base_t::uI_t uI_t;

    typedef std::tuple<typename EBs::Edge_t...> Edge_t;
    typedef std::tuple<typename EBs::Data_t...> Data_t;

    template <typename E> static constexpr bool is_Edge_type
        = Sycl_Graph::is_Edge_type<E> && has_type<E, Edge_t>::value;
    static constexpr uI_t invalid_id = std::numeric_limits<uI_t>::max();

    template <Edge_type E> auto get_edges() const { return get_buffer<E>().get_edges(); }

    template <typename V> auto get_edges(const std::vector<uI_t> &ids) const {
      return get_buffer<V>().get_edges(ids);
    }

    auto get_edges() const { return std::make_tuple((get_buffer<EBs>().get_edges(), ...)); }
  };

  template <Sycl_Graph::Edge_Buffer_type... EBs>
  struct Edge_Buffer_Pack<Sycl_Graph::Buffer_Pack, EBs...>;

  template <typename T>
  concept Edge_Buffer_Pack_type = true;
}  // namespace Sycl_Graph
#endif