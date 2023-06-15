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
namespace Sycl_Graph {

  template <typename ... Ts>
  struct Vertex;

  template <typename D>
  struct Vertex<D> {
    typedef uint32_t ID_t;
    static constexpr ID_t invalid_id = std::numeric_limits<uint32_t>::max();
    typedef D Data_t;
    Vertex() = default;

    Vertex(ID_t id, const D& data) : id(id), data(data) {}
    ID_t id = invalid_id;
    Data_t data;

    bool is_valid() const { return id != invalid_id; }
  };

  // template <typename T>
  // static constexpr bool is_Vertex_type = std::unsigned_integral<typename T::ID_t> &&
  // std::is_convertible_v<decltype(std::declval<T>().id), typename T::ID_t> &&
  // std::is_convertible_v<decltype(std::declval<T>().data), typename T::Data_t>;

  template <typename T>
  concept Vertex_type = std::unsigned_integral<typename T::ID_t>
                        && requires(T t) {
                             { t.id } -> std::convertible_to<typename T::ID_t>;
                             { t.data } -> std::convertible_to<typename T::Data_t>;
                           };

  template <typename T> constexpr bool is_Vertex_type = Vertex_type<T>;

  struct Connection_ID_Pair {
    typedef uint32_t ID_t;
    static constexpr ID_t invalid_id = std::numeric_limits<uint32_t>::max();
    ID_t from = invalid_id;
    ID_t to = invalid_id;

    bool is_valid() const { return to != invalid_id && from != invalid_id; }

    bool operator==(const Connection_ID_Pair& other) const {
      return to == other.to && from == other.from;
    }
  };

  template <typename ... Ts>
  struct Edge;

  template <typename D, Vertex_type _From_t, Vertex_type _To_t,
            typename _Connection_IDs>
  struct Edge<D, _From_t, _To_t, _Connection_IDs> {
    typedef D Data_t;
    typedef _From_t From_t;
    typedef _To_t To_t;
    typedef _Connection_IDs Connection_IDs;
    typedef typename Connection_IDs::ID_t ID_t;
    static constexpr auto invalid_id = Connection_IDs::invalid_id;
    Connection_IDs ids;
    Data_t data = Data_t{};

    Edge(Connection_IDs ids, const D& data) : data(data), ids(ids) {}
    Edge(ID_t from, ID_t to, const D& data) : data(data), ids{from, to} {}
    Edge(ID_t from, ID_t to) : ids{from, to} {}
    Edge() = default;
    ID_t& from = ids.from;
    ID_t& to = ids.to;

    Edge operator=(const Edge& other) {
      ids = other.ids;
      data = other.data;
      from = ids.from;
      to = ids.to;
      return *this;
    }

    bool is_valid() const { return from != invalid_id && to != invalid_id; }
  };

  enum Edge_Direction_t : uint8_t {
    Edge_Direction_Undirected = 0,
    Edge_Direciton_Directed = 1,
    Edge_Direction_Bidirectional = 2
  };

  template <typename T>
  concept Edge_type = requires(T t) {
                        typename T::Data_t;
                        typename T::Connection_IDs;
                        typename T::ID_t;
                        T::invalid_id;
                      };
  typedef Edge<void> ID_Edge_t;

  template <typename T> constexpr bool is_Edge_type = Edge_type<T>;

  template <typename T>
  concept Graph_element = Edge_type<T> || Vertex_type<T>;

  template <typename T> constexpr bool is_Graph_element = Graph_element<T>;

}  // namespace Sycl_Graph
#endif  // SYCL_GRAPH_GRAPH_TYPES_HPP
