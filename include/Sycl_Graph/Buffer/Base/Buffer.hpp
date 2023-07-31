#ifndef SYCL_GRAPH_BUFFER_BASE_HPP
#define SYCL_GRAPH_BUFFER_BASE_HPP
#include <Sycl_Graph/type_helpers.hpp>

namespace Sycl_Graph {
  template <typename T>
  concept Buffer_type = true;

  template <typename... Ds> struct Buffer {
    typedef std::tuple<Ds...> Data_t;
    static constexpr size_t N_buffers = sizeof...(Ds);
    static constexpr std::array<size_t, N_buffers> buffer_type_sizes = {sizeof(Ds)...};
    std::tuple<std::vector<Ds>...> bufs;
    uint32_t curr_size = 0;
    std::shared_ptr<spdlog::logger> logger;

    Buffer(uint32_t N,
           std::string logger_filename = "buffer_" + std::to_string(logging::Buffer_id++))
        : bufs(std::make_tuple(std::vector<Ds>(N))...),
          logger(spdlog::basic_logger_mt(logger_filename, logger_filename + ".log", true)) {
      logger->set_level(spdlog::level::debug);
      curr_size = N;
    }

    Buffer(sycl::queue &q, const std::vector<Ds> &...data, const sycl::property_list &props = {},
           std::string logger_filename = "buffer_" + std::to_string(logging::Buffer_id++))
        : q(q), logger(spdlog::basic_logger_mt(logger_filename, logger_filename + ".log", true)) {
      logger->set_level(spdlog::level::debug);
      this->bufs = std::make_tuple(data...);
      curr_size = std::get<0>(bufs)->size();
      logger->info("Buffer created with size {}", curr_size);
      std::string data_types = "";
      ((data_types += std::string(typeid(Ds).name()) + ", "), ...);
      logger->info("Buffer contains data types: {}", data_types);
    }

    uint32_t current_size() const { return curr_size; }

    template <typename T> static constexpr bool is_Data_type = (std::is_same_v<T, Ds> || ...);

    template <typename... Ts> static constexpr bool is_Data_types = (is_Data_type<Ts> && ...);

    template <typename T> static constexpr void Data_type_assert() {
      static_assert(is_Data_type<T>, "Invalid data type");
    }

    template <typename... Ts> static constexpr void Data_types_assert() {
      (Data_type_assert<Ts>(), ...);
    }

    auto get_buffers() const { return bufs; }

    template <typename D> auto &get_buffer() {
      static_assert((std::is_same_v<D, Ds> || ...), "Buffer does not contain type D");
      std::cout << "getting buffer of type " << typeid(D).name() << std::endl;
      return std::get<std::vector<D>>(bufs);
    }

    void resize(uint32_t new_size) {
      std::apply([&](auto &...args) { (args.resize(new_size), ...); }, bufs);
      curr_size = std::min(curr_size, new_size);
    }

    void add(const std::tuple<std::vector<Ds>...> &data) {
      std::apply(
          [&](auto &...args) {
            (args.insert(args.end(), std::get<std::vector<Ds>>(data).begin(),
                         std::get<std::vector<Ds>>(data).end()),
             ...);
          },
          bufs);
      curr_size += std::get<0>(data).size();
    }


    template <typename D> void remove_elements(bool (*cond)(const D &)) {
      auto N_removed = buffer_remove(bufs, q, cond);
      curr_size -= N_removed;
    }

    void remove_elements(uint32_t offset, uint32_t size) {
      auto N_removed = buffer_remove(bufs, q, offset, size);
      curr_size -= N_removed;
    }

    void remove_elements(const std::vector<uint32_t> &indices) {
      auto N_removed = buffer_remove(bufs, q, indices);
      curr_size -= N_removed;
    }

    Buffer<Ds...> &operator=(const Buffer<Ds...> &other) {
      bufs = other.bufs;
      return *this;
    }

    Buffer<Ds...> &operator+(const Buffer<Ds...> &other) {
        std::apply(
            [&](auto &...args) {
                (args.insert(args.end(), std::get<std::vector<Ds>>(other.bufs).begin(),
                             std::get<std::vector<Ds>>(other.bufs).end()),
                 ...);
            },
            bufs);
      return *this;
    }

    uint32_t current_size() { return curr_size; }

    uint32_t max_size() const { return std::get<0>(bufs).size(); }

    uint32_t byte_size() const {
      return std::accumulate(buffer_type_sizes.begin(), buffer_type_sizes.end(), 0) * max_size();
    }
  };

}  // namespace Sycl_Graph

#endif
