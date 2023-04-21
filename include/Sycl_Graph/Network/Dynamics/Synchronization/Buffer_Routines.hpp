#ifndef SYCL_GRAPH_INVARIANT_BUFFER_ROUTINES_HPP
#define SYCL_GRAPH_INVARIANT_BUFFER_ROUTINES_HPP
#include <Sycl_Graph/Edge_Buffer_Pack.hpp>
#include <Sycl_Graph/Vertex_Buffer_Pack.hpp>
#include <Sycl_Graph/type_helpers.hpp>
#include <array>
#include <map>
namespace Sycl_Graph::Network::Dynamics {

template <Sycl_Graph::Vertex_Buffer_type Vertex_Buffer_t, 
          Sycl_Graph::Edge_Buffer_type Edge_Buffer_t> requires std::is_same_v<typename Vertex_Buffer_t::Vertex_t, typename Edge_Buffer_t::Edge_t::To_t>
auto get_incoming_degrees(const Vertex_Buffer_t& vertex_buffer, const Edge_Buffer_t& edge_buffer)
{
    static_assert<
    Type_Map vertex_degrees(vertex_buffer.buffers);
}

template <Sycl_Graph::Vertex_Buffer_type Vertex_Buffer_t, 
          Sycl_Graph::Edge_Buffer_type Edge_Buffer_t> requires std::is_same_v<typename Vertex_Buffer_t::Vertex_t, typename Edge_Buffer_t::Edge_t::From_t>
Type_Map< get_outgoing_degrees(const Vertex_Buffer_t& vertex_buffer, const Edge_Buffer_t& edge_buffer)
{
    std::vector<Vertex_Buffer_t::uI_t> vertex_degrees(vertex_buffer.size());
    
    
}

template <Sycl_Graph::Vertex_Buffer_Pack_type Vertex_Buffer_t, 
          Sycl_Graph::Edge_Buffer_Pack_type Edge_Buffer_t>
auto get_vertex_degrees(const Vertex_Buffer_t& vertex_buffer, const Edge_Buffer_t& edge_buffer)
{
    Type_Map vertex_degrees(vertex_buffer.buffers);

}

//resize the contents of a buffer to match the size of another buffer
template <Sycl_Graph::Buffer_type Buffer_t, template <Buffer_t, Buffer_t> Matching_Condition>
void buffer_resize(const Buffer_t &source_buffer, 
                    Buffer_t& target_buffer) {
  std::apply(
      [&](const auto &...source_bufs) {
            std::apply([&](auto&... target_bufs){
      (target_bufs.resize(source_bufs.size()), ...);}, target_buffer.buffers);
      },
      source_buffer.buffers);
}

template <>
struct Vertex_Edge_Match
{
}

} // namespace Sycl_Graph::Network::Dynamics



#endif