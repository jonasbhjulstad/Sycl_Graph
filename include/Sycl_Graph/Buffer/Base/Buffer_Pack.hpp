#ifndef SYCL_GRAPH_GRAPH_INVARIANT_BUFFER_PACK_HPP
#define SYCL_GRAPH_GRAPH_INVARIANT_BUFFER_PACK_HPP
#include <Sycl_Graph/Buffer/Base/Buffer.hpp>
#include <Sycl_Graph/Graph/Base/Graph_Types.hpp>
#include <tuple>
namespace Sycl_Graph {
  template <Sycl_Graph::Buffer_type... Bs> struct Buffer_Pack
  {

    typedef std::tuple<Bs...> Buffer_t;
    typedef std::tuple<typename Bs::Data_t...> Data_t;
    static constexpr uint32_t N_buffers = sizeof...(Bs);
    Buffer_Pack() = default;
    Buffer_Pack(const Bs &...buffers) : buffers(std::make_tuple(buffers...)) {}
    Buffer_Pack(const Bs &&...buffers) : buffers(std::make_tuple(buffers...)) {}
    Buffer_Pack(const std::tuple<Bs...> &buffers) : buffers(buffers) {}
    Buffer_Pack(const Buffer_Pack &other) : buffers(other.buffers) {}

    typedef Buffer_Pack<Bs...> This_t;
    Buffer_t buffers;

    static constexpr uint32_t invalid_id = std::numeric_limits<uint32_t>::max();

    template <typename T> static constexpr bool is_Buffer_type = has_type<T, Buffer_t>::value;
    template <typename T> static constexpr bool is_Data_type = (has_type<T, typename Bs::Data_t>::value && ...);
    template <typename ... Ts> static constexpr bool is_Data_types = (is_Data_type<Ts> && ...);

    template <typename... Ds> auto &get_buffer() const {
      return std::get<index_of_type<std::tuple<Ds...>, typename Bs::Data_t...>()>(buffers);
    }

    auto size() const {
      return std::apply([](auto &&...buffers) { return (buffers.size() + ...); }, buffers);
    }

    template <typename D> auto &get_buffer(const uint32_t &id) const {
      return get_buffer<D>().get_buffer(id);
    }

    template <typename D> auto size() const { return get_buffer<D>().size(); }

    template <typename... Ds> void add(const std::vector<Ds> &&...data) {
      (get_buffers<Ds>().add(data), ...);
    }

    template <typename... Ds> void remove(const std::vector<Ds> &&...elements) {
      ((get_buffers<Ds>().remove(elements), ...));
    }

    auto &operator=(This_t &&other) {
      buffers = std::move(other.buffers);
      return *this;
    }

    auto copy() const {
      Buffer_Pack B;
      B.buffers = this->buffers;
      return B;
    }

    auto &operator+(const This_t &other) {
      std::apply(
          [&other](auto &&...buffers) { return std::make_tuple((buffers + other.buffers)...); },
          this->buffers);
      return *this;
    }

    template <typename D> void resize(const uint32_t &size) { get_buffer<D>().resize(size); }

    template <typename D> uint32_t current_size() const { return get_buffer<D>().current_size(); }

    uint32_t current_size() const {
      return std::apply([](auto... args) { return (args.current_size() + ...); }, buffers);
    }

    template <typename D> uint32_t max_size() const { return get_buffer<D>().max_size(); }
  };

  template <typename T>
  concept Buffer_Pack_type = Buffer_type<T> && requires(T t)
  {
    t.buffers;
    // T::Buffer_t;
  };
}  // namespace Sycl_Graph

#endif  // SYCL_GRAPH_GRAPH_INVARIANT_BUFFER_HPP
