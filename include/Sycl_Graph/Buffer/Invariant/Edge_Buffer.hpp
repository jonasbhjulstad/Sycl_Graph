#ifndef SYCL_GRAPH_INVARIANT_EDGE_BUFFER_HPP
#define SYCL_GRAPH_INVARIANT_EDGE_BUFFER_HPP
#include <Sycl_Graph/type_helpers.hpp>
#include <Sycl_Graph/Buffer/Invariant/Buffer.hpp>

namespace Sycl_Graph::Invariant
{    
    template <Sycl_Graph::Base::Edge_Buffer_type... EBs>
    struct Edge_Buffer: public Buffer<EBs...>
    {
        typedef Buffer<EBs...> Base_t;
        Edge_Buffer() = default;
        Edge_Buffer(const EBs &...buffers) : Base_t(buffers ...) {}
        Edge_Buffer(const EBs &&...buffers) : Base_t(buffers ...) {}


        typedef typename Base_t::uI_t uI_t;

        typedef std::tuple<typename EBs::Edge_t...> Edge_t;
        typedef std::tuple<typename EBs::Edge_t::Data_t...> Data_t;

        template <typename T>
        using Edge_Type_Maps = std::tuple<Type_Map<T, typename EBs::Edge_t>...>;

        template <typename T>
        static constexpr bool is_Edge_type = std::disjunction_v<std::is_same<T, typename EBs::Edge_t>...>;
        static constexpr uI_t invalid_id = std::numeric_limits<uI_t>::max();

        template <Edge_type E>
        auto get_edges() const
        {
            return get_buffer<E>().get_edges();
        }

        template <typename V>
        auto get_edges(const std::vector<uI_t> &ids) const
        {
            return get_buffer<V>().get_edges(ids);
        }

        auto get_edges() const
        {
            return std::make_tuple((get_buffer<EBs>().get_edges(), ...));
        }
    };

    template <typename T>
    concept Edge_Buffer_type = true;
}
#endif