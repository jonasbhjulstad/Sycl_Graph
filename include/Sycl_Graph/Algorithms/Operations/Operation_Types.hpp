#ifndef SYCL_GRAPH_ALGORITHMS_PROPERTIES_OPERATION_TYPES_HPP
#define SYCL_GRAPH_ALGORITHMS_PROPERTIES_OPERATION_TYPES_HPP
#include <CL/sycl.hpp>
#include <Sycl_Graph/Graph/Base/Graph_Types.hpp>
#include <Sycl_Graph/Buffer/Sycl/Vertex_Buffer.hpp>
#include <Sycl_Graph/Buffer/Sycl/Edge_Buffer.hpp>
#include <concepts>
#include <memory>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>
#include <type_traits>
namespace Sycl_Graph::Sycl
{

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
template <typename T>
constexpr bool is_Transform_Operation_type = Transform_Operation_type<T>;

template <typename T>
concept Injection_Operation_type = Operation_type<T> && requires(T op) {
    is_Graph_element<typename T::Target_t>;
    typename T::Target_t;
};
template <typename T>
constexpr bool is_Injection_Operation_type = Injection_Operation_type<T>;

template <typename T>
concept Extraction_Operation_type = Operation_type<T> && requires(T op) {
    is_Graph_element<typename T::Source_t>;
    typename T::Source_t;
};
template <typename T>
constexpr bool is_Extraction_Operation_type = Extraction_Operation_type<T>;
template <typename T>
concept has_Source = requires(T op) { typename T::Source_t; };

template <typename T>
bool constexpr has_Source_v = has_Source<T>;

template <typename T>
concept has_Target = requires(T op) {
    typename T::Target_t;
    //check that method target_buffer_size exists
};

template <typename T>
bool constexpr has_Target_v = has_Target<T>;

template <Operation_type Op>
constexpr bool is_transform = has_Source_v<Op> && has_Target_v<Op>;
template <Operation_type Op>
constexpr bool is_injection = has_Source_v<Op> && !has_Target_v<Op>;
template <Operation_type Op>
constexpr bool is_extraction = !has_Source_v<Op> && has_Target_v<Op>;
template <Operation_type Op>
constexpr bool is_inplace_modification = !has_Source_v<Op> && !has_Target_v<Op>;


struct Operation_Buffer_Void_t
{
    char dummy;
};


namespace logging
{



template <sycl::access_mode mode, Vertex_type Vertex_t>
void log_accessor(std::shared_ptr<spdlog::logger> logger, Vertex_Accessor<mode, Vertex_t>& v_acc)
{
    logger->debug("Vertex_Accessor\ttype: {}\tcount: {}", typeid(Vertex_t).name(), v_acc.size());
}

template <sycl::access_mode mode, Edge_type Edge_t>
void log_accessor(std::shared_ptr<spdlog::logger> logger, Edge_Accessor<mode, Edge_t>& e_acc)
{
    logger->debug("Edge_Accessor\ttype: {}\tcount: {}", typeid(Edge_t).name(), e_acc.size());
}
template <typename DataT,
          int Dimensions,
          sycl::access::mode AccessMode,
          sycl::access::target AccessTarget,
          sycl::access::placeholder IsPlaceholder,
          typename PropertyListT>
void log_accessor(auto &logger,
                  sycl::accessor<DataT, Dimensions, AccessMode, AccessTarget, IsPlaceholder, PropertyListT> &acc)
{
    logger->debug("Accessor\ttype: {}\tcount: {}", typeid(DataT).name(), acc.size());
}
template <typename ... Ts>
void log_accessor(std::shared_ptr<spdlog::logger> logger,  std::tuple<Ts ...>& accessors)
{
    //create index sequence
    std::index_sequence_for<decltype(accessors)> seq;
    std::apply(
        [&](auto &&...acc) {
            (log_accessor(logger, acc), ...);
        },
        accessors);
}

static size_t N_Operation_Loggers = 0;
static uint32_t N_Global_Operation_Invocations = 0;
static std::string Global_Operation_Log_File_Name = "Operation_Debug.log";
static std::shared_ptr<spdlog::logger> Global_Operation_Logger =
    spdlog::basic_logger_mt("Global_Operation_Debug_Logger", Global_Operation_Log_File_Name, true);
} // namespace logging

template <typename T, sycl::access::mode _mode>
struct Accessor_t
{
    typedef T type;
    static constexpr sycl::access::mode mode = _mode;
};

template <typename T>
concept Accessor_type = true;

template <typename T>
concept Accessor_types = tuple_like<T>;


template <typename Derived, Accessor_types Accessors_t>
struct Operation_Base
{
    size_t ID;
    size_t N_invocations = 0;
    std::shared_ptr<spdlog::logger> logger;
    std::shared_ptr<spdlog::logger> global_logger;




    auto get_access(sycl::handler& h, Graph_type auto& graph, tuple_like auto& custom_buffers)
    {

        return std::apply([&](const auto ... acc)
        {
            return std::make_tuple(get_access<decltype(acc)>(h, graph, custom_buffers, acc.mode) ...);
        }, Accessors_t{});
    }

    template <typename T>
    auto get_access(sycl::handler& h, Graph_type auto& graph, tuple_like auto& custom_buffers, sycl::access::mode mode)
    {
        if constexpr (is_Vertex_type<T> || is_Edge_type<T>)
        {
            return graph.template get_access<T, mode>(h);
        }
        else
        {
            return std::get<T>(custom_buffers).get_access<mode>(h);
        }
    }

    Operation_Base(std::shared_ptr<spdlog::logger> logger)
        : logger(logger), global_logger(logging::Global_Operation_Logger)
    {
    }
    Operation_Base(const char *name, const char *log_file_name)
        : ID(logging::N_Operation_Loggers++), global_logger(logging::Global_Operation_Logger)
    {
        logger = spdlog::basic_logger_mt(name, log_file_name, true);
        logger->flush_on(spdlog::level::debug);
    }
    Operation_Base() : ID(logging::N_Operation_Loggers++), global_logger(logging::Global_Operation_Logger)
    {
        logger = spdlog::basic_logger_mt("operation_logger_" + std::to_string(ID),
                                         std::string("Operation_Debug_") + std::to_string(ID) + std::string(".log"), true);
        logger->flush_on(spdlog::level::debug);
    }

    template <typename... Acc>
    void __invoke(sycl::handler &h, Graph_type auto& graph, const tuple_like auto&& source_bufs, tuple_like auto&& target_bufs, tuple_like auto&& custom_bufs)
    {
        logger->debug("Global invocation {}, local invocation {}",
                      logging::N_Global_Operation_Invocations++,
                      N_invocations++);
        global_logger->debug("Global invocation {}, local invocation {}",
                             logging::N_Global_Operation_Invocations,
                             N_invocations);

        auto [source_accs, target_accs] = get_access(h, graph, source_bufs, target_bufs, custom_bufs);
        log_accessors(source_accs, target_accs);


        return std::apply([&](const auto&& ... source_acc)
        {
            return std::apply([&](auto&& ... target_acc)
            {
                return static_cast<Derived*>(this)->invoke(h, source_acc ..., target_acc ...);
            }, target_accs);
        }, source_accs);
    }

    private:

    void log_accessors(const tuple_like auto& source_accs, const tuple_like auto& target_accs) const
    {
        std::apply([&](auto &&...acc) { (logging::log_accessor(logger, acc), ...); }, source_accs);
        std::apply([&](auto &&...acc) { (logging::log_accessor(logger, acc), ...); }, target_accs);
        //global
        std::apply([&](auto &&...acc) { (logging::log_accessor(global_logger, acc), ...); }, source_accs);
        std::apply([&](auto &&...acc) { (logging::log_accessor(global_logger, acc), ...); }, target_accs);
    }

};

template <typename Derived, Accessor_type First, Accessor_type... Rest>
struct Operation_Base<Derived, std::tuple<First, Rest ...>>;


} // namespace Sycl_Graph::Sycl

#endif
