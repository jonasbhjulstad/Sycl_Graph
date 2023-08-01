#ifndef SYCL_GRAPH_ALGORITHMS_PROPERTIES_SYCL_PROPERTY_EXTRACTOR_HPP
#define SYCL_GRAPH_ALGORITHMS_PROPERTIES_SYCL_PROPERTY_EXTRACTOR_HPP

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

#include <CL/sycl.hpp>
#include <Sycl_Graph/Algorithms/Operations/Edge_Operations.hpp>
#include <Sycl_Graph/Algorithms/Operations/Operation_Buffers.hpp>
#include <Sycl_Graph/Algorithms/Operations/Operation_Types.hpp>
#include <Sycl_Graph/Algorithms/Operations/Vertex_Operations.hpp>
#include <Sycl_Graph/Buffer/Sycl/type_helpers.hpp>
#include <Sycl_Graph/Graph/Sycl/Graph.hpp>
#include <array>
#include <tuple>

namespace Sycl_Graph::Sycl {

  template <Operation_type Op>
  sycl::event invoke_operation(Graph_type auto &graph, Op &operation,
                               const tuple_type auto &source_bufs, tuple_type auto &target_bufs,
                               tuple_type auto &custom_bufs, auto &dep_event) {
    return graph.q.submit([&](sycl::handler &h) {
      h.depends_on(dep_event);
      operation.__invoke(h, graph, source_bufs, target_bufs, custom_bufs);
    });
  }

  template <Operation_type... Op>
  auto invoke_operations(Graph_type auto &graph, std::tuple<Op...> &operations,
                         tuple_type auto &source_bufs, tuple_type auto &target_bufs,
                         tuple_type auto &custom_bufs,
                         UniformTuple<sizeof...(Op), sycl::event> dep_events
                         = UniformTuple<sizeof...(Op), sycl::event>{}) {
    static_assert(!std::is_same_v<decltype(std::get<0>(target_bufs)), std::nullptr_t>);

    auto shuffled_tuples
        = shuffle_tuples(operations, source_bufs, target_bufs, custom_bufs, dep_events);
    std::apply([&](auto &&...tup) { return std::make_tuple(invoke_operation(graph, tup)...); },
               shuffled_tuples);
  }
  template <Operation_type... Op>
  auto invoke_operations(Graph_type auto &graph, std::tuple<Op...> &operations,
                         tuple_type auto &source_bufs, tuple_type auto &target_bufs,
                         UniformTuple<sizeof...(Op), sycl::event> dep_events
                         = UniformTuple<sizeof...(Op), sycl::event>{}) {
    static_assert(!std::is_same_v<decltype(std::get<0>(target_bufs)), std::nullptr_t>);

    auto shuffled_tuples = shuffle_tuples(operations, source_bufs, target_bufs,
                                          EmptyTuple<sizeof...(Op)>{}, dep_events);
    std::apply([&](auto &&...tup) { return std::make_tuple(invoke_operation(graph, tup)...); },
               shuffled_tuples);
  }

  template <Operation_type... Op>
  auto invoke_operation_sequence(Graph_type auto &graph, std::tuple<Op...> &operations,
                                 tuple_type auto &source_bufs, tuple_type auto &target_bufs,
                                 tuple_type auto &custom_bufs,
                                 sycl::event dep_event = sycl::event{});

  template <Operation_type... Op>
  void verify_operation_input_dimensions(std::tuple<Op...> &operations,
                                         tuple_type auto &source_bufs, tuple_type auto &target_bufs,
                                         tuple_type auto &custom_bufs) {
    static constexpr size_t N_ops = std::tuple_size_v<std::tuple<Op...>>;
    // check that source_bufs, target_bufs and custom_bufs have the same size
    static_assert(std::tuple_size_v<std::remove_reference_t<decltype(source_bufs)>> == N_ops);
    static_assert(std::tuple_size_v<std::remove_reference_t<decltype(target_bufs)>> == N_ops);
    static_assert(std::tuple_size_v<std::remove_reference_t<decltype(custom_bufs)>> == N_ops);
    // check that target buffers are the same as source buffers for the next operation
    auto source_tail = tuple_tail(source_bufs);
    auto target_head = drop_last_tuple_elem(target_bufs);

    std::apply(
        [&](auto &&...source) {
          std::apply(
              [&](auto &&...target) {
                static_assert((std::is_same_v<decltype(source), decltype(target)> && ...));
              },
              target_head);
        },
        source_tail);
  }

  template <Operation_type... Op>
  auto invoke_operation_sequence(Graph_type auto &graph, std::tuple<Op...> &operations,
                                 tuple_type auto &source_bufs, tuple_type auto &target_bufs,
                                 tuple_type auto &custom_bufs, sycl::event dep_event) {
    auto event = invoke_operation(graph, std::get<0>(operations), std::get<0>(source_bufs),
                                  std::get<0>(target_bufs), std::get<0>(custom_bufs), dep_event);
    if constexpr (std::tuple_size_v<std::tuple<Op...>> > 1) {
      auto other_events
          = invoke_operation_sequence(graph, tuple_tail(operations), tuple_tail(source_bufs),
                                      tuple_tail(target_bufs), tuple_tail(custom_bufs), event);
      return std::tuple_cat(std::make_tuple(event), other_events);
    } else {
      return std::make_tuple(event);
    }
  }

  template <Operation_type... Op>
  auto invoke_operation_sequence(Graph_type auto &graph, std::tuple<Op...> &operations,
                                 tuple_type auto &source_bufs, tuple_type auto &target_bufs,
                                 sycl::event dep_event) {
    EmptyTuple<sizeof...(Op)> custom_bufs;
    return invoke_operation_sequence(graph, operations, source_bufs, target_bufs, custom_bufs,
                                     dep_event);
  }
  template <Operation_type... Op>
  auto invoke_operation_sequence(Graph_type auto &graph, std::tuple<Op...> &&operations,
                                 tuple_type auto &&source_bufs, tuple_type auto &&target_bufs,
                                 tuple_type auto &&custom_bufs, sycl::event dep_event) {
    // verify_operation_input_dimensions(operations, source_bufs, target_bufs, custom_bufs);

    auto event = invoke_operation(graph, std::get<0>(operations), std::get<0>(source_bufs),
                                  std::get<0>(target_bufs), std::get<0>(custom_bufs), dep_event);
    if constexpr (sizeof...(Op) > 1) {
      auto other_events
          = invoke_operation_sequence(graph, tuple_tail(operations), tuple_tail(source_bufs),
                                      tuple_tail(target_bufs), tuple_tail(custom_bufs), event);
      return std::tuple_cat(std::make_tuple(event), other_events);
    } else {
      return std::make_tuple(event);
    }
  }



  template <Operation_type... Op>
  auto invoke_operation_sequence(Graph_type auto &graph, std::tuple<Op...> &&operations,
                                 tuple_type auto &&buffers, sycl::event dep_event) {
    return invoke_operation_sequence(graph, operations, std::get<0>(buffers), std::get<1>(buffers),
                                     std::get<2>(buffers), dep_event);
  }

  template <typename Derived, Accessor_type... Acc_Ts> struct Operation_Base {
    size_t ID;
    size_t N_invocations = 0;
    std::shared_ptr<spdlog::logger> logger;
    std::shared_ptr<spdlog::logger> global_logger;

    typedef std::tuple<Accessor_t<typename Acc_Ts::type, Acc_Ts::mode>...> Accessors_t;

    static constexpr Accessors_t accessors = Accessors_t{};
    static constexpr std::array<sycl::access::mode, sizeof...(Acc_Ts)> access_modes
        = {Acc_Ts::mode...};

    std::tuple<std::pair<typename Acc_Ts::type, size_t>...> _size_map;

    template <std::size_t I> void set_size(size_t size) { std::get<I>(_size_map).second = size; }

    template <std::size_t I> size_t get_size() { return std::get<I>(_size_map).second; }
    template <typename T> void set_size(size_t size) {
      std::get<std::pair<T, size_t>>(_size_map).second = size;
    }

    template <typename T> size_t get_size() {
      return std::get<std::pair<T, size_t>>(_size_map).second;
    }

    Operation_Base(const char *name, const char *log_file_name)
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

    void __invoke(sycl::handler &h, Graph_type auto &graph, tuple_type auto &&bufs) {
      logger->debug("Global invocation {}, local invocation {}",
                    logging::N_Global_Operation_Invocations++, N_invocations++);
      global_logger->debug("Global invocation {}, local invocation {}",
                           logging::N_Global_Operation_Invocations, N_invocations);
      auto accessors = operation_buffer_access<Acc_Ts::mode...>(h, bufs);

      log_accessors(accessors);

      std::apply([&](auto &&...acc) { static_cast<Derived *>(this)->invoke(h, acc...); },
                 accessors);
    }

    template <typename ... Ts>
    void __invoke(sycl::handler& h, Ts& ... accs)
    {
      logger->debug("Global invocation {}, local invocation {}",
                    logging::N_Global_Operation_Invocations++, N_invocations++);
      global_logger->debug("Global invocation {}, local invocation {}",
                           logging::N_Global_Operation_Invocations, N_invocations);
      log_accessors(std::make_tuple(accs...));
      static_cast<Derived *>(this)->invoke(h, accs...);
    }

    void __invoke(sycl::handler &h, Graph_type auto &graph, tuple_type auto &source_bufs,
                  tuple_type auto &target_bufs, tuple_type auto &custom_bufs) {
      this->__invoke(h, graph, std::tuple_cat(source_bufs, target_bufs, custom_bufs));
    }

    // private:
    void log_accessors(const tuple_type auto &accessors) const {
      std::apply([&](auto &&...acc) { (logging::log_accessor(logger, acc), ...); }, accessors);
      // global
      std::apply([&](auto &&...acc) { (logging::log_accessor(global_logger, acc), ...); },
                 accessors);
    }

    // template <Accessor_type Acc_t>
    // auto create_buffers(const Graph_type auto &graph, const Acc_t &acc) const {
    //   using Acc_type = typename Acc_t::type;
    //   if constexpr (is_Vertex_type<Acc_type> || is_Edge_type<Acc_type>) {
    //     return graph.template get_buffer<Acc_type>();
    //   } else {
    //     size_t size = this->template get_size<Acc_type>();
    //     return std::make_shared<sycl::buffer<Acc_type>>(
    //         sycl::buffer<Acc_type>((sycl::range<1>(size))));
    //   }
    // }

    // auto create_buffers(const Graph_type auto &graph) const {
    //   auto bufs = std::apply(
    //       [&](auto &&...acc) {
    //         return std::make_tuple(this->template create_buffer<obt>(graph, acc)...);
    //       },
    //       accessors);
    //   return tuple_filter<std::shared_ptr<void>>(bufs);
    // }
  };

}  // namespace Sycl_Graph::Sycl
#endif
