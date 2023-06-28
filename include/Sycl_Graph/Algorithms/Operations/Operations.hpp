#ifndef SYCL_GRAPH_ALGORITHMS_PROPERTIES_SYCL_PROPERTY_EXTRACTOR_HPP
#define SYCL_GRAPH_ALGORITHMS_PROPERTIES_SYCL_PROPERTY_EXTRACTOR_HPP

#include <CL/sycl.hpp>
#include <Sycl_Graph/Algorithms/Operations/Edge_Operations.hpp>
#include <Sycl_Graph/Algorithms/Operations/Operation_Buffers.hpp>
#include <Sycl_Graph/Algorithms/Operations/Operation_Types.hpp>
#include <Sycl_Graph/Algorithms/Operations/Vertex_Operations.hpp>
#include <Sycl_Graph/Buffer/Sycl/type_helpers.hpp>
#include <Sycl_Graph/Graph/Sycl/Graph.hpp>
#include <array>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>
#include <tuple>

namespace Sycl_Graph::Sycl
{

namespace _detail
{
template <Graph_type Graph_t, Operation_type Op, std::size_t... Indices>
auto get_graph_accessor_impl(Graph_t &graph, sycl::handler &h, const std::index_sequence<Indices...> &)
{
    return std::make_tuple(graph.template get_access<std::get<Indices>(Op::graph_access_modes),
                                                     std::tuple_element_t<Indices, typename Op::Accessor_Types>>(h)...);
}
template <Graph_type Graph_t, Operation_type Op, std::size_t... Indices>
auto get_custom_accessor_impl(const auto &buf)
{
}

} // namespace _detail
template <Graph_type Graph_t, Operation_type Op>
auto get_graph_accessors(Graph_t &graph, sycl::handler &h)
{
    // create index sequence for 'typename Op::Accessor_Types'
    constexpr size_t size = std::tuple_size<typename Op::Accessor_Types>::value;
    constexpr auto seq = std::make_integer_sequence<size_t, size>();
    return _detail::get_graph_accessor_impl<Graph_t, Op>(graph, h, seq);
}

template <Graph_type Graph_t, Operation_type Op>
auto get_custom_accessors(tuple_like auto& bufs, sycl::handler &h)
{
    static_assert(std::tuple_size_v<std::remove_reference_t<decltype(bufs)>> == Op::custom_access_modes.size(),
                  "Number of access modes and accessor types must be equal");
    if constexpr (std::tuple_size_v<decltype(Op::custom_access_modes)> == 0)
    {
        return std::tuple<>();
    }
    else
    {
        return std::apply(
            [&](auto &&...buf) {
                return std::apply(
                    [&](auto... mode) {
                        assert((std::is_same_v<decltype(mode), sycl::access::mode> && ...));
                        assert(((mode == sycl::access_mode::atomic) && ...));
                        return std::make_tuple(buf->template get_access<sycl::access_mode::atomic>(h)...);
                    },
                    Op::custom_access_modes);
            },
            bufs);
    }
}
template <Operation_type Op>
sycl::event invoke_operation(Graph_type auto &graph,
                             Op &operation,
                             auto&& source_buf,
                             auto&& target_buf,
                             tuple_like auto& custom_bufs,
                             auto &dep_event)
{
    return graph.q.submit([&](sycl::handler &h) {
        h.depends_on(dep_event);
        auto graph_accessors =
            get_graph_accessors<decltype(graph), std::remove_reference_t<decltype(operation)>>(graph, h);
        constexpr bool is_custom_buf = std::tuple_size_v<std::remove_reference_t<decltype(custom_bufs)>> > 0;
        if constexpr (is_transform<Op>)
        {

            auto source_acc = source_buf->template get_access<sycl::access_mode::read>(h);
            auto target_acc = target_buf->template get_access<sycl::access_mode::write>(h);

            if constexpr (is_custom_buf)
            {
                auto custom_accessors =
                    get_custom_accessors<decltype(graph), std::remove_reference_t<decltype(operation)>>(custom_bufs, h);
                operation.__invoke(h, graph_accessors, custom_accessors, source_acc, target_acc);
            }
            else
            {
                operation.__invoke(h, graph_accessors, source_acc, target_acc);
            }
        }
        else if constexpr (is_injection<Op>)
        {

            auto source_acc = source_buf->template get_access<sycl::access_mode::read>(h);
            if constexpr (is_custom_buf)
            {
                auto custom_accessors =
                    get_custom_accessors<decltype(graph), std::remove_reference_t<decltype(operation)>>(custom_bufs, h);
                operation.__invoke(h, graph_accessors, custom_accessors, source_acc);
            }
            else
            {
                operation.__invoke(h, graph_accessors, source_acc);
            }
        }
        else if constexpr (is_extraction<Op>)
        {

            auto target_acc = target_buf->template get_access<sycl::access_mode::write>(h);

            if constexpr (is_custom_buf)
            {
                auto custom_accessors =
                    get_custom_accessors<decltype(graph), std::remove_reference_t<decltype(operation)>>(custom_bufs, h);
                operation.__invoke(h, graph_accessors, custom_accessors, target_acc);
            }
            else
            {
                operation.__invoke(h, graph_accessors, target_acc);
            }
        }
        else if constexpr (is_inplace_modification<Op>)
        {

            if constexpr (is_custom_buf)
            {
                auto custom_accessors =
                    get_custom_accessors<decltype(graph), std::remove_reference_t<decltype(operation)>>(custom_bufs, h);

                operation.__invoke(h, graph_accessors, custom_accessors);
            }
            else
            {
                operation.__invoke(h, graph_accessors);
            }
        }
    });
}

sycl::event invoke_operation(Graph_type auto &G, tuple_like auto &&tuple)
{
    return std::apply([&](auto &&...tup) { return invoke_operation(G, tup...); });
}


template <Operation_type... Op>
auto invoke_operations(Graph_type auto &graph,
                       std::tuple<Op...> &operations,
                       tuple_like auto& source_bufs,
                       tuple_like auto& target_bufs,
                       tuple_like auto& custom_bufs,
                       UniformTuple<sizeof...(Op), sycl::event> dep_events = UniformTuple<sizeof...(Op), sycl::event>{})
{

    static_assert(
        !std::is_same_v<decltype(std::get<0>(target_bufs)), std::nullptr_t>);

    auto shuffled_tuples = shuffle_tuples(operations, source_bufs, target_bufs, custom_bufs, dep_events);
    std::apply([&](auto &&...tup) { return std::make_tuple(invoke_operation(graph, tup)...); }, shuffled_tuples);
}
template <Operation_type... Op>
auto invoke_operations(Graph_type auto &graph,
                       std::tuple<Op...> &operations,
                       tuple_like auto& source_bufs,
                       tuple_like auto& target_bufs,
                       UniformTuple<sizeof...(Op), sycl::event> dep_events = UniformTuple<sizeof...(Op), sycl::event>{})
{

    static_assert(
        !std::is_same_v<decltype(std::get<0>(target_bufs)), std::nullptr_t>);

    auto shuffled_tuples = shuffle_tuples(operations, source_bufs, target_bufs, EmptyTuple<sizeof...(Op)>{}, dep_events);
    std::apply([&](auto &&...tup) { return std::make_tuple(invoke_operation(graph, tup)...); }, shuffled_tuples);
}


template <Operation_type... Op>
auto invoke_operation_sequence(Graph_type auto &graph,
                               std::tuple<Op...> &operations,
                               tuple_like auto& source_bufs,
                               tuple_like auto& target_bufs,
                               tuple_like auto& custom_bufs,
                               sycl::event dep_event = sycl::event{});

template <Operation_type... Op>
auto invoke_operation_sequence(Graph_type auto &graph,
                               std::tuple<Op ...> &operations,
                               tuple_like auto& source_bufs,
                               tuple_like auto& target_bufs,
                               tuple_like auto& custom_bufs,
                               sycl::event dep_event)
{
    auto event = invoke_operation(graph,
                                  std::get<0>(operations),
                                  std::get<0>(source_bufs),
                                  std::get<0>(target_bufs),
                                  std::get<0>(custom_bufs),
                                  dep_event);
    if constexpr (std::tuple_size_v<std::tuple<Op...>> > 1)
    {
        auto other_events = invoke_operation_sequence(graph,
                                                      tuple_tail(operations),
                                                      tuple_tail(source_bufs),
                                                      tuple_tail(target_bufs),
                                                      tuple_tail(custom_bufs),
                                                      event);
        return std::tuple_cat(std::make_tuple(event), other_events);
    }
    else
    {
        return std::make_tuple(event);
    }
}

template <Operation_type ... Op>
auto invoke_operation_sequence(Graph_type auto &graph,
                               std::tuple<Op...> &&operations,
                               tuple_like auto&& source_bufs,
                               tuple_like auto&& target_bufs,
                               tuple_like auto&& custom_bufs,
                               sycl::event dep_event)
{
    auto event = invoke_operation(graph,
                                  std::get<0>(operations),
                                  std::get<0>(source_bufs),
                                  std::get<0>(target_bufs),
                                  std::get<0>(custom_bufs),
                                  dep_event);
#ifdef OPERATION_DEBUG_TARGET_BUFS
    auto target_vec = buffer_get(std::get<0>(target_bufs));
#endif
    int a = 0;
    if constexpr (std::tuple_size_v<std::tuple<Op...>> > 1)
    {
        auto other_events = invoke_operation_sequence(graph,
                                                      tuple_tail(operations),
                                                      tuple_tail(source_bufs),
                                                      tuple_tail(target_bufs),
                                                      tuple_tail(custom_bufs),
                                                      event);
        return std::tuple_cat(std::make_tuple(event), other_events);
    }
    else
    {
        return std::make_tuple(event);
    }
}


} // namespace Sycl_Graph::Sycl
#endif
