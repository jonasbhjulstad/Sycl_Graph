#include <iostream>
#include <tuple>
#include <type_traits>
#include <typeinfo>

template <typename... Ts>
auto operation(std::tuple<Ts...> &args)
{
    return std::apply([](auto &...elem) { return (elem + ...); }, args);
}

template <std::size_t I>
using index_t = std::integral_constant<std::size_t, I>;

template <std::size_t... Is>
using indexes_t = std::tuple<index_t<Is>...>;

template <std::size_t... Is>
constexpr indexes_t<Is...> to_indexes(std::index_sequence<Is...>)
{
    return {};
}

template <class... Ts>
constexpr auto to_indexes(std::tuple<Ts...> const &)
{
    return to_indexes(std::make_index_sequence<sizeof...(Ts)>{});
}

template <class Tuple, class F>
constexpr auto map_tuple(Tuple &&tuple, F f)
{
    return std::apply([&](auto &&...ts) { return std::make_tuple(f(ts)...); }, std::forward<Tuple>(tuple));
}


constexpr std::tuple<> shuffle_tuples()
{
    return {};
}


template <class T0, class... Tuples>
constexpr auto shuffle_tuples(T0 &&t0, Tuples &&...tuples)
{
    return map_tuple(to_indexes(t0), [&](auto I) {
        return std::make_tuple(std::get<I>(std::forward<T0>(t0)), std::get<I>(std::forward<Tuples>(tuples))...);
    });
}
void print_tuple(auto &&tup)
{
        std::apply([&](auto&&... t)
        {
            ((std::cout << t << " "), ...);
        }, tup);
}

int main()
{
    std::tuple<int, float, double> Foo{1, 10.f, 100.0};
    std::tuple<float, bool, char> Bar{2.0f, true, '2'};

    //...

    std::tuple<char, float, char> Baz{'3', 3.f, '3'};


    auto shuffled_tuples = shuffle_tuples(Foo, Bar, Baz);
    std::apply([](auto &&...tup) { ((print_tuple(tup), std::cout << '\n'), ...); }, shuffled_tuples);

    auto result = std::apply([&](auto&&... st)
    {
        return std::make_tuple(operation(st)...);
    }, shuffled_tuples);

    std::cout << "Result:\t";

    print_tuple(result);
    std::cout << "\n";

    return 0;
}
