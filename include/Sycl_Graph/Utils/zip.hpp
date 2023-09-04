#ifndef SYCL_GRAPH_UTILS_ZIP_HPP
#define SYCL_GRAPH_UTILS_ZIP_HPP
#include <vector>
#include <tuple>
namespace Sycl_Graph
{
    namespace detail
    {
    template <typename ... Ts>
    auto zip_impl(std::tuple<std::vector<Ts> ...>&& ts)
    {
        auto N = std::get<0>(ts).size();
        std::vector<std::tuple<Ts ...>> result(N);
        std::generate(result.begin(), result.end(), [&ts, i = 0]() mutable
        {
            return std::apply([&i](auto& ... xs) mutable
            {
                return std::make_tuple(xs[i] ...);
            }, ts);
        });
        return result;
    }
    } //detail

    template <typename ... Ts>
    auto zip(const std::vector<Ts>& ... vs)
    {
        return detail::zip_impl(std::make_tuple(vs ...));
    }
} //Sycl_Graph

#endif
