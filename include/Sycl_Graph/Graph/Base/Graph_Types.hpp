#ifndef SYCL_GRAPH_GRAPH_TYPES_HPP
#define SYCL_GRAPH_GRAPH_TYPES_HPP
#include <Sycl_Graph/Graph/Base/Graph_Types.hpp>
#include <Sycl_Graph/Math/math.hpp>
#include <Sycl_Graph/type_helpers.hpp>
#include <concepts>
#include <limits>
#include <numeric>
#include <ostream>
#include <vector>
namespace Sycl_Graph
{

template <typename D>
struct Vertex
{
    static constexpr uint32_t invalid_id = std::numeric_limits<uint32_t>::max();
    typedef D Data_t;
    Vertex() = default;
    struct ID_t
    {
        uint32_t value = invalid_id;
        bool operator!=(const ID_t& id_1)
        {
            return this->value != id_1.value;
        }
    };
    Vertex(uint32_t id, const D &data) : id{id}, data(data)
    {
    }
    ID_t id;
    Vertex(uint32_t id): id{id}{}
    Data_t data;

    bool is_valid() const
    {
        return id.value != invalid_id;
    }
};

template <>
struct Vertex<void>
{
    static constexpr uint32_t invalid_id = std::numeric_limits<uint32_t>::max();
    typedef void Data_t;
    struct ID_t
    {
        uint32_t value = invalid_id;
        bool operator!=(const ID_t& id_1)
        {
            return this->value != id_1.value;
        }
        //make ID_t convertible to uint32_t
        operator uint32_t() const
        {
            return this->value;
        }
    };

    Vertex() = default;

    Vertex(uint32_t id) : id{id}
    {
    }
    ID_t id;
    bool is_valid() const
    {
        return id.value != invalid_id;
    }
};

typedef Vertex<void> Void_Vertex_t;

template <typename T>
concept Vertex_type = requires(T t) {
    T::id.value;
    typename T::ID_t;
};

template <typename T>
constexpr bool is_Vertex_type = Vertex_type<T>;


template <typename D, Vertex_type _From_t = Void_Vertex_t, Vertex_type _To_t = Void_Vertex_t>
struct Edge
{
    typedef D Data_t;
    typedef _From_t From_t;
    typedef _To_t To_t;
    static constexpr auto invalid_id = std::numeric_limits<uint32_t>::max();
    Data_t data = Data_t{};

    struct ID_Pair_t
    {
      static constexpr uint32_t invalid_id = std::numeric_limits<uint32_t>::max();
        uint32_t from = invalid_id;
        uint32_t to = invalid_id;
    };
    ID_Pair_t id;
    Edge(uint32_t from, uint32_t to, const D &data) : id{from, to}, data(data)
    {
    }

    Edge(ID_Pair_t id, const D& data): id(id), data(data){}

    Edge(uint32_t from, uint32_t to): id{from, to}{}

    Edge() = default;

    Edge operator=(const Edge &other)
    {
        id = other.id;
        data = other.data;
        return *this;
    }

    bool is_valid() const
    {
        return id.from != invalid_id && id.to != invalid_id;
    }
};


enum Edge_Direction_t : uint8_t
{
    Edge_Direction_Undirected = 0,
    Edge_Direciton_Directed = 1,
    Edge_Direction_Bidirectional = 2
};

template <typename T>
concept Edge_type = requires(T t) {
    typename T::Data_t;
    T::id;
    T::invalid_id;
    T::id.to;
    T::id.from;
};
typedef Edge<void> Void_Edge_t;

template <typename T>
constexpr bool is_Edge_type = Edge_type<T>;

template <typename T>
concept Graph_element = Edge_type<T> || Vertex_type<T>;

template <typename T>
constexpr bool is_Graph_element = Graph_element<T>;

} // namespace Sycl_Graph
#endif // SYCL_GRAPH_GRAPH_TYPES_HPP
