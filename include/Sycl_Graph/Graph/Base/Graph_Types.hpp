#ifndef SYCL_GRAPH_GRAPH_TYPES_HPP
#define SYCL_GRAPH_GRAPH_TYPES_HPP
#include <limits>
#include <concepts>
#include <numeric>
#include <vector>
#include <Sycl_Graph/Graph/Graph_Types.hpp>
#include <Sycl_Graph/type_helpers.hpp>
#include <Sycl_Graph/Math/math.hpp>
#include <ostream>
namespace Sycl_Graph::Base
{

    template <typename D, typename _ID_t = uint32_t, _ID_t _invalid_id = std::numeric_limits<_ID_t>::max()>
    struct Vertex
    {
        typedef _ID_t ID_t;
        static constexpr ID_t invalid_id = _invalid_id;
        typedef D Data_t;
        
        Vertex(ID_t id, const D& data): id(id), data(data) {}        
        ID_t id = invalid_id;
        Data_t data;

        bool is_valid() const
        {
            return id != invalid_id;
        }

    };

    template <typename T>
    concept Vertex_type = 
    std::unsigned_integral<typename T::ID_t> &&
    requires(T t)
    {
        {t.id} -> std::convertible_to<typename T::ID_t>;
        {t.data} -> std::convertible_to<typename T::Data_t>;
    };

    template <typename _ID_t = uint32_t, _ID_t _invalid_id = std::numeric_limits<_ID_t>::max()>
    struct Connection_ID_Pair
    {
        typedef _ID_t ID_t;
        _ID_t to = invalid_id;
        _ID_t from = invalid_id;
        static constexpr ID_t invalid_id = _invalid_id;

        bool is_valid() const
        {
            return to != invalid_id && from != invalid_id;
        }

        bool operator==(const Connection_ID_Pair &other) const
        {
            return to == other.to && from == other.from;
        }
    };
    template <typename D, typename _Connection_IDs = Connection_ID_Pair<>>
    struct Edge
    {
        typedef D Data_t;
        typedef _Connection_IDs Connection_IDs;
        typedef typename Connection_IDs::ID_t ID_t;
        static constexpr auto invalid_id = Connection_IDs::invalid_id;
        Data_t data = Data_t{};
        Edge(const D &data, ID_t to, ID_t from)
            : data(data), ids{to, from} {}
        Edge(ID_t to, ID_t from)
            : ids{to, from} {}
        Connection_IDs ids;
    };


    template <typename T>
    concept Edge_type = requires(T t)
    {
        typename T::Data_t;
        typename T::ID_t;
        T::invalid_id;
    };
} // namespace Sycl_Graph
#endif // SYCL_GRAPH_GRAPH_TYPES_HPP