#ifndef SYCL_GRAPH_ALGORITHMS_PROPERTIES_OPERATION_BUFFERS_HPP
#define SYCL_GRAPH_ALGORITHMS_PROPERTIES_OPERATION_BUFFERS_HPP
#include <Sycl_Graph/Algorithms/Operations/Operation_Types.hpp>
#include <memory>
#include <type_traits>
namespace Sycl_Graph::Sycl
{


template <Graph_type Graph_t, Operation_type Op>
auto create_source_buffer(const Graph_t &G, const Op &op)
{
    if constexpr (has_Source_v<Op>)
    {
        return std::make_shared<sycl::buffer<typename Op::Source_t>>(
            sycl::buffer<typename Op::Source_t>(op.source_buffer_size(G)));
    }
    else
    {
        return nullptr;
    }
}


template <Graph_type Graph_t, Operation_type Op>
auto create_target_buffer(const Graph_t &G, const Op &op)
{
    static_assert(has_Target_v<Op>, "Operation does not have a target buffer");
    if constexpr (has_Target_v<Op>)
        return std::make_shared<sycl::buffer<typename Op::Target_t>>(
            sycl::buffer<typename Op::Target_t>(sycl::range<1>(op.target_buffer_size(G))));
    else
        return nullptr;
}


template <Graph_type Graph_t, Operation_type Op>
auto create_operation_buffers(const Graph_t &G, const std::tuple<Op> &operation)
{
    return std::make_pair(create_source_buffer(G, operation), create_target_buffer(G, operation));
}


template <Graph_type Graph_t, Operation_type... Op, typename Buf_Prev_t>
auto create_operation_buffer_sequence(const Graph_t &G,
                                      const std::tuple<Op...> &operations,
                                      Buf_Prev_t target_buffer_prev);


template <Graph_type Graph_t, Operation_type... Op, typename Buf_Prev_t>
auto create_operation_buffer_sequence(const Graph_t &G,
                                      const std::tuple<Op...> &operations,
                                      Buf_Prev_t target_buffer_prev)
{
    typedef std::tuple_element_t<0, std::tuple<Op...>> First;
    auto& first_op = std::get<0>(operations);
    auto op_tail = drop_first_tuple_elem(operations);
    constexpr size_t op_size = std::tuple_size_v<std::tuple<Op...>>;
    auto target_buf = create_target_buffer(G, first_op);

    if constexpr (has_Source_v<First>)
    {
        static_assert(std::is_same_v<typename First::Source_t, typename Buf_Prev_t::element_type::value_type>);
        if constexpr (op_size <= 1)
        {
            return std::make_pair(std::tuple_cat(std::make_tuple(target_buffer_prev)),
                                  std::tuple_cat(std::make_tuple(target_buf)));
        }
        else
        {
            auto [source_buf_tail, target_buf_tail] = create_operation_buffer_sequence(G, op_tail, target_buf);
            return std::make_pair(std::tuple_cat(std::make_tuple(target_buffer_prev), source_buf_tail),
                                  std::tuple_cat(std::make_tuple(target_buf), target_buf_tail));
        }
    }
    else
    {


        auto source_buf = create_source_buffer(G, first_op);
        if constexpr (op_size <= 1)
        {
            return std::make_pair(std::tuple_cat(std::make_tuple(source_buf), std::make_tuple(target_buffer_prev)),
                                  std::tuple_cat(std::make_tuple(target_buf)));
        }
        else
        {

            auto [source_buf_tail, target_buf_tail] = create_operation_buffer_sequence(G, op_tail, target_buf);
            return std::make_pair(std::tuple_cat(std::make_tuple(source_buf), source_buf_tail),
                                  std::tuple_cat(std::make_tuple(target_buf), target_buf_tail));
        }
    }
}

template <Graph_type Graph_t, Operation_type... Op>
auto create_operation_buffer_sequence(const Graph_t &G, const std::tuple<Op...> &operations)
{
    auto& first_op = std::get<0>(operations);
    auto source_buf = create_source_buffer(G, first_op);
    auto target_buf = create_target_buffer(G, first_op);

    if constexpr (std::tuple_size_v<std::remove_reference_t<decltype(operations)>> <= 1)
    {
        return std::make_pair(std::make_tuple(source_buf), std::make_tuple(target_buf));
    }
    else
    {
        auto op_tail = drop_first_tuple_elem(operations);
        auto [source_buf_tail, target_buf_tail] = create_operation_buffer_sequence(G, op_tail, target_buf);

        return std::make_pair(std::tuple_cat(std::make_tuple(source_buf), source_buf_tail),
                              std::tuple_cat(std::make_tuple(target_buf), target_buf_tail));
    }
}


} // namespace Sycl_Graph::Sycl
#endif
