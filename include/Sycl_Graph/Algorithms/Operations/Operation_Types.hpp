#ifndef SYCL_GRAPH_ALGORITHMS_PROPERTIES_OPERATION_TYPES_HPP
#define SYCL_GRAPH_ALGORITHMS_PROPERTIES_OPERATION_TYPES_HPP
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

#include <CL/sycl.hpp>
#include <Sycl_Graph/Buffer/Sycl/Edge_Buffer.hpp>
#include <Sycl_Graph/Buffer/Sycl/Vertex_Buffer.hpp>
#include <Sycl_Graph/Graph/Base/Graph_Types.hpp>
#include <concepts>
#include <memory>
#include <type_traits>
namespace Sycl_Graph::Sycl {

  template <typename T>
  concept Operation_type = true;

  namespace logging {

    template <sycl::access_mode mode, Vertex_type Vertex_t>
    void log_accessor(std::shared_ptr<spdlog::logger> logger,
                      const Vertex_Accessor<mode, Vertex_t>& v_acc) {
      logger->debug("Vertex_Accessor\ttype: {}\tcount: {}", typeid(Vertex_t).name(), v_acc.size());
    }

    template <sycl::access_mode mode, Edge_type Edge_t>
    void log_accessor(std::shared_ptr<spdlog::logger> logger,
                      const Edge_Accessor<mode, Edge_t>& e_acc) {
      logger->debug("Edge_Accessor\ttype: {}\tcount: {}", typeid(Edge_t).name(), e_acc.size());
    }
    template <typename DataT, int Dimensions, sycl::access::mode AccessMode,
              sycl::access::target AccessTarget, sycl::access::placeholder IsPlaceholder,
              typename PropertyListT>
    void log_accessor(auto& logger,
                      const sycl::accessor<DataT, Dimensions, AccessMode, AccessTarget,
                                           IsPlaceholder, PropertyListT>& acc) {
      logger->debug("Accessor\ttype: {}\tcount: {}", typeid(DataT).name(), acc.size());
    }
    template <typename... Ts>
    void log_accessor(std::shared_ptr<spdlog::logger> logger, std::tuple<Ts...>& accessors) {
      // create index sequence
      std::index_sequence_for<decltype(accessors)> seq;
      std::apply([&](auto&&... acc) { (log_accessor(logger, acc), ...); }, accessors);
    }

    static size_t N_Operation_Loggers = 0;
    static uint32_t N_Global_Operation_Invocations = 0;
    static std::string Global_Operation_Log_File_Name = "Operation_Debug.log";
    static std::shared_ptr<spdlog::logger> Global_Operation_Logger = spdlog::basic_logger_mt(
        "Global_Operation_Debug_Logger", Global_Operation_Log_File_Name, true);
  }  // namespace logging

  template <typename T, sycl::access::mode _mode>
  struct Accessor_t {
    typedef T type;
    static constexpr sycl::access::mode mode = _mode;
  };

  template <typename... Ts> using Write_Accessors_t
      = std::tuple<Accessor_t<Ts, sycl::access_mode::write>...>;

  template <typename... Ts> using ReadWrite_Accessors_t
      = std::tuple<Accessor_t<Ts, sycl::access_mode::read_write>...>;

  template <typename... Ts> using Read_Accessors_t
      = std::tuple<Accessor_t<Ts, sycl::access_mode::read>...>;

  template <typename... Ts> using Atomic_Accessors_t
      = std::tuple<Accessor_t<Ts, sycl::access_mode::atomic>...>;

  template <typename T>
  concept Accessor_type = true;

  template <typename T>
  concept Accessor_types = true;

  template <Accessor_type T> static constexpr bool is_target_accessor
      = T::mode == sycl::access_mode::atomic || T::mode == sycl::access_mode::write
        || T::mode == sycl::access_mode::read_write
        || T::mode == sycl::access_mode::discard_read_write;

  template <Accessor_type T> static constexpr bool is_source_accessor
      = T::mode == sycl::access_mode::read;

}  // namespace Sycl_Graph::Sycl

#endif
