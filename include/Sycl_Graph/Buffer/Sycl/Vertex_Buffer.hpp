#ifndef SYCL_GRAPH_BUFFER_SYCL_VERTEX_BUFFER_HPP
#define SYCL_GRAPH_BUFFER_SYCL_VERTEX_BUFFER_HPP
#include <CL/sycl.hpp>
#include <Sycl_Graph/Buffer/Base/Vertex_Buffer.hpp>
#include <Sycl_Graph/Buffer/Sycl/Buffer.hpp>
#include <Sycl_Graph/Buffer/Sycl/Buffer_Routines.hpp>
#include <concepts>
#include <iostream>
#include <type_traits>
namespace Sycl_Graph::Sycl
{

template <typename T>
auto get_vertex_data_accessor(const auto& acc)
{
    if constexpr (is_Vertex_type<T>)
    {
        return acc.data;
    }
    else
    {
        return acc;
    }
}

template <sycl::access::mode Mode, Sycl_Graph::Vertex_type Vertex_t>
struct Vertex_Accessor : public Buffer_Accessor<Mode, typename Vertex_t::ID_t, typename Vertex_t::Data_t>
{
    typedef typename Vertex_t::Data_t Data_t;
    typedef typename Vertex_t::ID_t ID_t;
    typedef Buffer_Accessor<Mode, ID_t, Data_t> Base_t;
    Vertex_Accessor(Base_t &&base) : Base_t(base)
    {
    }
    typedef Vertex_t value_type;

    sycl::accessor<ID_t, 1, Mode> ids = std::get<0>(this->accessors);
    sycl::accessor<Data_t, 1, Mode> data = std::get<1>(this->accessors);
    Vertex_t operator[](uint32_t idx) const
    {
        auto [id, data] = this->get_idx(idx);
        return Vertex_t(id, data);
    }

    size_t size() const
    {
        return this->ids.size();
    }
};


template <Sycl_Graph::Vertex_type Vertex_t>
auto vertex_to_vectors(const std::vector<Vertex_t> &vertices)
{
    //reserve
    auto data = std::vector<typename Vertex_t::Data_t>(vertices.size());
    auto ids = std::vector<typename Vertex_t::ID_t>(vertices.size());
    for (size_t i = 0; i < vertices.size(); i++)
    {
        ids[i] = vertices[i].id;
        data[i] = vertices[i].data;
    }

    return std::make_tuple(ids, data);
}

namespace logging
{
static size_t N_Vertex_Buffers = 0;
}
template <Sycl_Graph::Vertex_type _Vertex_t>
struct Vertex_Buffer : public Buffer<typename _Vertex_t::ID_t, typename _Vertex_t::Data_t>
{

    typedef Buffer<typename _Vertex_t::ID_t, typename _Vertex_t::Data_t> Base_t;
    typedef _Vertex_t Vertex_t;
    typedef typename _Vertex_t::ID_t ID_t;
    typedef typename Base_t::Data_t Data_t;
    typedef typename Vertex_t::Data_t Vertex_Data_t;
    sycl::queue &q = Base_t::q;
    Vertex_Buffer(sycl::queue &q, uint32_t NV = 1, const sycl::property_list &props = {}) : Base_t(q, NV, props)
    {
    }

    Vertex_Buffer(sycl::queue &q,
                  const std::vector<ID_t> &ids,
                  const std::vector<Vertex_Data_t> &data = {},
                  const sycl::property_list &props = {})
        : Base_t(q, ids, data, props)
    {
        this->bufs = std::make_tuple(buffer_initialize_shared(ids), buffer_initialize_shared(data));
    }

    template <typename T>
    static constexpr bool has_Vertex_type()
    {
        return std::is_same_v<T, Vertex_t>;
    }


    std::vector<uint32_t> get_valid_ids()
    {
        auto &id_buf = this->template get_buffer<uint32_t>();
        std::vector<uint32_t> ids = buffer_get(id_buf);
        ids.erase(std::remove_if(ids.begin(), ids.end(), [](const uint32_t &id) { return id == Vertex_t::invalid_id; }),
                  ids.end());
        return ids;
    }


    void add(const std::vector<Vertex_t> &vertices)
    {
        auto [ids, data] = vertex_to_vectors(vertices);
        using Buffer_Data_t = std::tuple_element_t<1, typename Base_t::Data_t>;
        static_assert(std::is_same_v<Buffer_Data_t, typename Vertex_t::Data_t> &&
                      "Data type mismatch between entry and buffer");
        auto tup = std::make_tuple(ids, data);
        static_cast<Base_t *>(this)->add(tup);
    }

    uint32_t N_vertices() const
    {
        return this->current_size();
    }

    std::vector<Vertex_t> get_vertices()
    {
        auto Vertex_tuple = this->template get<uint32_t, Data_t>();
        std::vector<Vertex_t> vertices;
        vertices.reserve(Vertex_tuple.first.size());
        for (size_t i = 0; i < Vertex_tuple.first.size(); i++)
        {
            vertices.push_back(Vertex_t(Vertex_tuple.first[i], Vertex_tuple.second[i]));
        }
        return vertices;
    }


    template <sycl::access_mode Mode>
    Vertex_Accessor<Mode, Vertex_t> get_access(sycl::handler &h)
    {
        return Vertex_Accessor<Mode, Vertex_t>(
            static_cast<Base_t *>(this)->template get_access<Mode, typename Vertex_t::ID_t, typename Vertex_t::Data_t>(
                h));
    }

    void remove(const std::vector<uint32_t> &ids)
    {
        this->template remove_elements<uint32_t>(ids);
    }

};

template <Sycl_Graph::Vertex_type Vertex_t>
auto make_vertex_buffer(sycl::queue &q, const std::vector<Vertex_t> &vertices, const sycl::property_list &props = {})
{
    auto [ids, data] = vertex_to_vectors(vertices);
    return std::make_shared<Vertex_Buffer<Vertex_t>>(Vertex_Buffer<Vertex_t>(q, ids, data, props));
}

template <typename T>
concept Vertex_Buffer_type = Sycl_Graph::Vertex_Buffer_type<T>;

} // namespace Sycl_Graph::Sycl

#endif // SYCL_GRAPH_SYCL_VERTEX_BUFFER_HPP
