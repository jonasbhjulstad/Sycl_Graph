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

  static constexpr int OPERATION_SIZE_INHERITED = -1;
  template <typename T>
  concept has_Graph_Iterator = true;

  template <typename T>
  concept Operation_type = true;

  template <typename T>
  concept Transform_Operation_type = Operation_type<T> && requires(T op) {
    typename T::Source_t;
    typename T::Target_t;
    typename T::target_access_mode;
  };
  template <typename T> constexpr bool is_Transform_Operation_type = Transform_Operation_type<T>;

  template <typename T>
  concept Injection_Operation_type = Operation_type<T> && requires(T op) {
    is_Graph_element<typename T::Target_t>;
    typename T::Target_t;
  };
  template <typename T> constexpr bool is_Injection_Operation_type = Injection_Operation_type<T>;

  template <typename T>
  concept Extraction_Operation_type = Operation_type<T> && requires(T op) {
    is_Graph_element<typename T::Source_t>;
    typename T::Source_t;
  };
  template <typename T> constexpr bool is_Extraction_Operation_type = Extraction_Operation_type<T>;
  template <typename T>
  concept has_Source = requires(T op) { typename T::Source_t; };

  template <typename T> bool constexpr has_Source_v = has_Source<T>;

  template <typename T>
  concept has_Target = requires(T op) {
    typename T::Target_t;
    // check that method target_buffer_size exists
  };

  template <typename T> bool constexpr has_Target_v = has_Target<T>;

  template <Operation_type Op> constexpr bool is_transform = has_Source_v<Op> && has_Target_v<Op>;
  template <Operation_type Op> constexpr bool is_injection = has_Source_v<Op> && !has_Target_v<Op>;
  template <Operation_type Op> constexpr bool is_extraction = !has_Source_v<Op> && has_Target_v<Op>;
  template <Operation_type Op> constexpr bool is_inplace_modification
      = !has_Source_v<Op> && !has_Target_v<Op>;

  struct Operation_Buffer_Void_t {
    char dummy;
  };

  template <typename Op, typename... Args>
  concept op_initialize_type
      = Operation_type<Op> && requires(Op op, Args... args) { op.initialize(args...); };

  template <typename Op, typename... Args> static constexpr bool has_initialize
      = op_initialize_type<Op, Args...>;

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

  template <typename T, sycl::access::mode _mode> struct Accessor_t {
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
  template <typename Derived, Accessor_types _Source_Accessors_t,
            Accessor_types _Target_Accessors_t, Accessor_types _Custom_Accessors_t = std::tuple<>>
  struct Operation_Base {
    size_t ID;
    size_t N_invocations = 0;
    std::shared_ptr<spdlog::logger> logger;
    std::shared_ptr<spdlog::logger> global_logger;
    typedef _Source_Accessors_t Source_Accessors_t;
    typedef _Target_Accessors_t Target_Accessors_t;
    typedef _Custom_Accessors_t Custom_Accessors_t;

    auto get_access(sycl::handler& h, Graph_type auto& graph, const tuple_type auto& source_bufs,
                    tuple_type auto& target_bufs, tuple_type auto& custom_bufs) {
      auto source_access = std::apply(
          [&](const auto... acc) {
            return std::make_tuple(get_access<decltype(acc)>(h, graph, source_bufs)...);
          },
          Source_Accessors_t{});

      auto target_access = std::apply(
          [&](const auto... acc) {
            return std::make_tuple(get_access<decltype(acc)>(h, graph, target_bufs)...);
          },
          Target_Accessors_t{});

      auto custom_access = std::apply(
          [&](const auto... acc) {
            return std::make_tuple(get_access<decltype(acc)>(h, graph, custom_bufs)...);
          },
          Custom_Accessors_t{});

      return std::make_tuple(source_access, target_access, custom_access);
    }

    template <Accessor_type Acc_t>
    auto get_access(sycl::handler& h, Graph_type auto& graph, const tuple_type auto& buffers) {
      using T = typename Acc_t::type;
      if constexpr (is_Vertex_type<T> || is_Edge_type<T>) {
        return graph.template get_access<Acc_t::mode, T>(h);
      } else {
        return std::get<std::shared_ptr<sycl::buffer<T>>>(buffers)
            ->template get_access<Acc_t::mode>(h);
      }
    }

    Operation_Base(std::shared_ptr<spdlog::logger> logger)
        : logger(logger), global_logger(logging::Global_Operation_Logger) {}
    Operation_Base(const char* name, const char* log_file_name)
        : ID(logging::N_Operation_Loggers++), global_logger(logging::Global_Operation_Logger) {
      logger = spdlog::basic_logger_mt(name, log_file_name, true);
      logger->flush_on(spdlog::level::debug);
    }
    Operation_Base()
        : ID(logging::N_Operation_Loggers++), global_logger(logging::Global_Operation_Logger) {
      logger = spdlog::basic_logger_mt(
          "operation_logger_" + std::to_string(ID),
          std::string("Operation_Debug_") + std::to_string(ID) + std::string(".log"), true);
      logger->flush_on(spdlog::level::debug);
    }
    template <typename... Args> void initialize(sycl::handler& h, const Args&... args) {
      std::cout << "Op does not have initialize" << std::endl;
    }

    template <typename... Acc, tuple_type Custom_Bufs_t>
    std::enable_if_t<(std::tuple_size_v < Custom_Bufs_t >> 0), void> _invoke(
        sycl::handler& h, Graph_type auto& graph, const tuple_type auto& source_bufs,
        tuple_type auto& target_bufs, Custom_Bufs_t& custom_bufs) {
      auto accessors = get_access(h, graph, source_bufs, target_bufs, custom_bufs);
      auto& source_accs = std::get<0>(accessors);
      auto& target_accs = std::get<1>(accessors);
      auto& custom_accs = std::get<2>(accessors);
      std::apply(
          [&](auto&... custom_acc) {
            std::apply(
                [&](const auto&... source_acc) {
                  std::apply(
                      [&](auto&... target_acc) {
                        static_cast<Derived*>(this)->invoke(source_acc..., target_acc...,
                                                            custom_acc..., h);
                      },
                      target_accs);
                },
                source_accs);
          },
          custom_accs);
    }

    template <typename... Acc, tuple_type Custom_Bufs_t>
    std::enable_if_t<std::tuple_size_v<Custom_Bufs_t> == 0, void> _invoke(
        sycl::handler& h, Graph_type auto& graph, const tuple_type auto& source_bufs,
        tuple_type auto& target_bufs, Custom_Bufs_t& custom_bufs) {
      auto accessors = get_access(h, graph, source_bufs, target_bufs, custom_bufs);
      auto& source_accs = std::get<0>(accessors);
      auto& target_accs = std::get<1>(accessors);
      // invocation
      std::apply(
          [&](const auto&... source_acc) {
            std::apply(
                [&](auto&... target_acc) {
                  static_cast<Derived*>(this)->invoke(source_acc..., target_acc..., h);
                },
                target_accs);
          },
          source_accs);
    }

    template <typename... Acc, typename ... Custom_Buf_Ts>
    void __initialize(
        sycl::handler& h, Graph_type auto& graph, const tuple_type auto& source_bufs,
        tuple_type auto& target_bufs, std::tuple<Custom_Buf_Ts ...>& custom_bufs) {
      auto accessors = get_access(h, graph, source_bufs, target_bufs, custom_bufs);
      auto& source_accs = std::get<0>(accessors);
      auto& target_accs = std::get<1>(accessors);
      auto& custom_accs = std::get<2>(accessors);
      std::apply(
          [&](auto&... custom_acc) {
            std::apply(
                [&](const auto&... source_acc) {
                  std::apply(
                      [&](auto&... target_acc) {
                        static_cast<Derived*>(this)->initialize(h, source_acc..., target_acc...,
                                                                custom_acc...);
                      },
                      target_accs);
                },
                source_accs);
          },
          custom_accs);
    }

    template <typename... Acc>
    void __initialize(
        sycl::handler& h, Graph_type auto& graph, const tuple_type auto& source_bufs,
        tuple_type auto& target_bufs, const std::tuple<>& custom_bufs) {
      auto accessors = get_access(h, graph, source_bufs, target_bufs, custom_bufs);
      auto& source_accs = std::get<0>(accessors);
      auto& target_accs = std::get<1>(accessors);
      // invocation
      std::apply(
          [&](const auto&... source_acc) {
            std::apply(
                [&](auto&... target_acc) {
                  static_cast<Derived*>(this)->initialize(h, source_acc..., target_acc...);
                },
                target_accs);
          },
          source_accs);
    }

    template <typename... Acc>
    void __invoke(sycl::handler& h, Graph_type auto& graph, const tuple_type auto& source_bufs,
                  tuple_type auto& target_bufs, tuple_type auto& custom_bufs) {
      logger->debug("Global invocation {}, local invocation {}",
                    logging::N_Global_Operation_Invocations++, N_invocations++);
      global_logger->debug("Global invocation {}, local invocation {}",
                           logging::N_Global_Operation_Invocations, N_invocations);
      _invoke(h, graph, source_bufs, target_bufs, custom_bufs);
    }

    // private:
    void log_accessors(const tuple_type auto& source_accs, const tuple_type auto& target_accs,
                       const tuple_type auto& custom_accs = std::tuple<>{}) const {
      std::apply([&](auto&&... acc) { (logging::log_accessor(logger, acc), ...); }, source_accs);
      std::apply([&](auto&&... acc) { (logging::log_accessor(logger, acc), ...); }, target_accs);
      // global
      std::apply([&](auto&&... acc) { (logging::log_accessor(global_logger, acc), ...); },
                 source_accs);
      std::apply([&](auto&&... acc) { (logging::log_accessor(global_logger, acc), ...); },
                 target_accs);

      if constexpr (std::tuple_size_v<std::remove_reference_t<decltype(custom_accs)>> > 0) {
        std::apply([&](auto&&... acc) { (logging::log_accessor(logger, acc), ...); }, custom_accs);
        std::apply([&](auto&&... acc) { (logging::log_accessor(global_logger, acc), ...); },
                   custom_accs);
      }
    }

    auto get_source_sizes(const Graph_type auto& graph) const {
      return _get_buffer_sizes<Source_Accessors_t>(graph);
    }

    auto get_target_sizes(const Graph_type auto& graph) const {
      return _get_buffer_sizes<Target_Accessors_t>(graph);
    }

    auto get_custom_sizes(const Graph_type auto& graph) const {
      return _get_buffer_sizes<Custom_Accessors_t>(graph);
    }

    template <Accessor_types Accs_t> auto _get_buffer_sizes(const Graph_type auto& graph) const {
      return std::apply(
          [&](const auto... acc) {
            return std::make_tuple(
                _get_buffer_size<typename std::remove_reference_t<decltype(acc)>::type>(graph)...);
          },
          Accs_t{});
    }
    template <typename T> size_t _get_buffer_size(const Graph_type auto& graph) const {
      if constexpr (is_Vertex_type<T> || is_Edge_type<T>)
        return graph.template current_size<T>();
      else
        return static_cast<const Derived*>(this)->template get_buffer_size<T>(graph);
    }
  };

}  // namespace Sycl_Graph::Sycl

#endif
