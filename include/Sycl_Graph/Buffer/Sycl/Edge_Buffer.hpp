#ifndef SYCL_GRAPH_SYCL_EDGE_BUFFER_HPP
#define SYCL_GRAPH_SYCL_EDGE_BUFFER_HPP
#include <CL/sycl.hpp>
#include <Sycl_Graph/Buffer/Base/Edge_Buffer.hpp>
#include <Sycl_Graph/Buffer/Sycl/Buffer.hpp>
#include <Sycl_Graph/Buffer/Sycl/Buffer_Routines.hpp>
#include <Sycl_Graph/Graph/Base/Graph_Types.hpp>
namespace Sycl_Graph::Sycl
{
template <sycl::access::mode Mode, Sycl_Graph::Edge_type Edge_t>
struct Edge_Accessor : public Buffer_Accessor<Mode, typename Edge_t::ID_Pair_t, typename Edge_t::Data_t>
{
    typedef typename Edge_t::ID_Pair_t ID_Pair_t;
    typedef typename Edge_t::Data_t Data_t;
    typedef Buffer_Accessor<Mode, ID_Pair_t, Data_t> Base_t;
    Edge_Accessor(const Base_t &base) : Base_t(base)
    {
    }
    typedef Edge_t value_type;
    sycl::accessor<ID_Pair_t, 1, Mode> ids = std::get<0>(this->accessors);
    sycl::accessor<Data_t, 1, Mode> data = std::get<1>(this->accessors);
    Edge_t operator[](sycl::id<1> idx) const
    {
        const auto &id = this->ids[idx];
        const auto &data = this->data[idx];
        return Edge_t(id, data);
    }

    size_t size() const
    {
        return this->ids.size();
    }
};

template <Sycl_Graph::Edge_type _Edge_t>
struct Edge_Buffer : public Buffer<typename _Edge_t::ID_Pair_t, typename _Edge_t::Data_t>
{
    typedef _Edge_t Edge_t;
    typedef typename _Edge_t::ID_Pair_t ID_Pair_t;
    typedef typename _Edge_t::Data_t Edge_Data_t;
    typedef Buffer<ID_Pair_t, Edge_Data_t> Base_t;
    typedef uint32_t uint32_t;

    sycl::queue &q = Base_t::q;
    Edge_Buffer(sycl::queue &q, uint32_t NE = 1, const sycl::property_list &props = {}) : Base_t(q, NE, props)
    {
    }

    Edge_Buffer(sycl::queue &q, const std::vector<ID_Pair_t>& ids, const std::vector<Edge_Data_t>& data, const sycl::property_list &props = {})
        : Base_t(q, ids, data, props)
    {
        this->bufs = std::make_tuple(buffer_initialize_shared(ids), buffer_initialize_shared(data));
    }


    std::vector<ID_Pair_t> get_valid_ids()
    {
        auto &id_buf = this->template get_buffer<ID_Pair_t>();
        std::vector<ID_Pair_t> ids = buffer_get(id_buf);
        ids.erase(std::remove_if(ids.begin(),
                                 ids.end(),
                                 [](const ID_Pair_t &id) {
                                     return id.to == ID_Pair_t::invalid_id ||
                                            id.from == ID_Pair_t::invalid_id;
                                 }),
                  ids.end());
        return ids;
    }

    uint32_t N_edges() const
    {
        return this->current_size();
    }

    void add(const std::vector<Edge_t> &edges)
    {
        assert(edges.size() > 0 && "Given empty edge list");
        std::vector<ID_Pair_t> ids;
        std::vector<typename Edge_t::Data_t> data;
        data.reserve(edges.size());
        ids.reserve(edges.size());
        for (const auto &e : edges)
        {
            ids.push_back(e.id);
            data.push_back(e.data);
        }
        static_cast<Base_t *>(this)->add(std::make_tuple(ids, data));
    }


    std::vector<Edge_t> get_edges()
    {
        std::vector<Edge_t> result(this->current_size());
        auto result_buf = sycl::buffer<Edge_t>(result.data(), result.size());

        this->q
            .submit([&](sycl::handler &h) {
                auto acc = this->template get_access<sycl::access_mode::read>(h);
                auto result_acc = result_buf.template get_access<sycl::access_mode::write>(h);
                h.parallel_for(sycl::range<1>(this->current_size()),
                               [=](sycl::id<1> idx) { result_acc[idx] = acc[idx]; });
            })
            .wait();
        return result;
    }

    void remove(const std::vector<ID_Pair_t> &ids)
    {
        this->template remove_elements<ID_Pair_t>(ids);
    }

    template <sycl::access_mode Mode>
    auto get_access(sycl::handler &h)
    {
        return Edge_Accessor<Mode, Edge_t>(std::move(
            static_cast<Base_t *>(this)->template get_access<Mode, ID_Pair_t, typename Edge_t::Data_t>(h)));
    }
};
template <Sycl_Graph::Edge_type Edge_t>
auto edge_to_vectors(const std::vector<Edge_t> &edges)
{
    //reserve
    std::vector<typename Edge_t::ID_Pair_t> ids(edges.size());
    std::vector<typename Edge_t::Data_t> data(edges.size());
    for (size_t i = 0; i < edges.size(); i++)
    {
        ids[i] = edges[i].id;
        data[i] = edges[i].data;
    }

    return std::make_tuple(ids, data);
}
template <Sycl_Graph::Edge_type Edge_t>
auto make_edge_buffer(sycl::queue &q, const std::vector<Edge_t> &edges, const sycl::property_list &props = {})
{
    auto [ids, data] = edge_to_vectors(edges);
    return Edge_Buffer<Edge_t>(q, ids, data, props);
}

template <typename T>
concept Edge_Buffer_type = Sycl_Graph::Edge_Buffer_type<T>;
} // namespace Sycl_Graph::Sycl

#endif //
