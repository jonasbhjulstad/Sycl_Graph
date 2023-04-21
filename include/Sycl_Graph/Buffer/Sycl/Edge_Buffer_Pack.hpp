#ifndef SYCL_GRAPH_SYCL_INVARIANT_EDGE_BUFFER_HPP
#define SYCL_GRAPH_SYCL_INVARIANT_EDGE_BUFFER_HPP
#include <Sycl_Graph/Buffer/Base/Edge_Buffer_Pack.hpp>
#include <Sycl_Graph/Buffer/Sycl/Edge_Buffer.hpp>
#include <Sycl_Graph/type_helpers.hpp>

namespace Sycl_Graph::Sycl {

  using Sycl_Graph::Sycl::Buffer_Pack;
  template <Sycl_Graph::Sycl::Edge_Buffer_type... EBs> struct Edge_Buffer_Pack
      : public Sycl_Graph::Edge_Buffer_Pack<Buffer_Pack, EBs...> {
    typedef Sycl_Graph::Edge_Buffer_Pack<Buffer_Pack, EBs...> Base_t;
    Edge_Buffer_Pack() = default;
    Edge_Buffer_Pack(const EBs &...buffers) : Base_t(buffers...) {}
    Edge_Buffer_Pack(const EBs &&...buffers) : Base_t(buffers...) {}

    typedef typename Base_t::uI_t uI_t;
    using Base_t::is_Edge_type;
    using typename Base_t::Data_t;
    using typename Base_t::Edge_t;

    static constexpr uI_t invalid_id = std::numeric_limits<uI_t>::max();

    template <Sycl_Graph::Edge_type E> auto get_edges() const {
      return get_buffer<E>().get_edges();
    }

    template <typename V> auto get_edges(const std::vector<uI_t> &ids) const {
      return get_buffer<V>().get_edges(ids);
    }

    auto get_edges() const { return std::make_tuple((get_buffer<EBs>().get_edges(), ...)); }

    template <sycl::access_mode Mode, typename E> auto get_access(sycl::handler &h) const {
      static_assert(Base_t::template is_Edge_type<E>);
      return this->template get_buffer<std::tuple<typename E::Connection_IDs, typename E::Data_t>>()
          .template get_access<Mode>(h);
    }
  };

  template <typename T>
  concept Edge_Buffer_Pack_type = true;
}  // namespace Sycl_Graph::Sycl
#endif