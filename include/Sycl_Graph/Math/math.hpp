#ifndef SYCL_GRAPH_Sycl_Graph_Math_HPP
#define SYCL_GRAPH_Sycl_Graph_Math_HPP

#include <algorithm>
#include <cstdint>
#include <vector>
#include <array>
#include <type_traits>
#include <stddef.h>
namespace Sycl_Graph
{

    template <typename T>
    std::vector<T> linspace(T min, T max, uint32_t N) {
        std::vector<T> res(N);
        for (int i = 0; i < N; i++) {
            res[i] = min + (max - min) * i / (N - 1);
        }
        return res;
    }

    constexpr auto n_choose_k(std::unsigned_integral auto n, std::unsigned_integral auto k)
    {
        return (k == 0) ? 1 : (n * n_choose_k(n - 1, k - 1)) / k;
    }
    template <typename T>
    std::vector<T> arange(T min, T max, T step)
    {
        T s = min;
        std::vector<T> res;
        while (s <= max)
        {
            res.push_back(s);
            s += step;
        }
        return res;
    }
    template <typename T = uint32_t>
    std::vector<T> range(auto start, auto end)
    {
        std::vector<T> res(end - start);
        for (T i = start; i < end; i++)
        {
            res[i - start] = i;
        }
        return res;
    }

    template <typename T>
    inline std::vector<std::pair<T, T>> zip(const std::vector<T> &a, const std::vector<T> &b)
    {
        std::vector<std::pair<T, T>> res(a.size());
        for (int i = 0; i < a.size(); i++)
        {
            res[i] = std::make_pair(a[i], b[i]);
        }
        return res;
    }

    template <typename T>
    inline std::pair<std::vector<T>, std::vector<T>> unzip(const std::vector<std::pair<T, T>> &a)
    {
        std::pair<std::vector<T>, std::vector<T>> res;
        res.first.reserve(a.size());
        res.second.reserve(a.size());
        for (int i = 0; i < a.size(); i++)
        {
            res.first.push_back(a[i].first);
            res.second.push_back(a[i].second);
        }
        return res;
    }

    template <typename T>
    inline std::vector<std::vector<T>> transpose(const std::vector<std::vector<T>> &vec)
    {
        // transpose vector
        std::vector<std::vector<T>> res(vec[0].size());
        for (int i = 0; i < vec[0].size(); i++)
        {
            res[i].reserve(vec.size());
            for (int j = 0; j < vec.size(); j++)
            {
                res[i].push_back(vec[j][i]);
            }
        }
        return res;
    }

    template <typename T, std::size_t N0, std::size_t N1>
    inline std::array<std::array<T, N1>, N0> transpose(const std::array<std::array<T, N0>, N1> &vec)
    {
        std::array<std::array<T, N1>, N0> res;
        for (int i = 0; i < N1; i++)
        {
            for (int j = 0; j < N0; j++)
            {
                res[j][i] = vec[i][j];
            }
        }
        return res;
    }

    template <uint32_t N, typename T, typename T_out = T>
    std::array<T_out, N - 1> diff(const std::array<T, N> &vec)
    {
        std::array<T_out, N - 1> res;
        for (int i = 0; i < N; i++)
        {
            res[i] = (T_out)vec[i + 1] - (T_out)vec[i];
        }
        return res;
    }

    template <uint32_t N, typename T, typename T_out = T>
    std::array<T_out, N> integrate(const T v0, const std::array<T, N - 1> &vec)
    {
        std::array<T_out, N> res;
        res[0] = v0;
        for (int i = 0; i < N; i++)
        {
            res[i + 1] = res[i] + vec[i];
        }
        return res;
    }

    template <typename T>
    std::vector<std::pair<uint32_t, T>> enumerate(const std::vector<T> &data)
    {
        std::vector<std::pair<uint32_t, T>> enumerated_data(data.size());
        std::transform(data.begin(), data.end(), enumerated_data.begin(), [&, n = -1](const auto &d) mutable
                       { return std::make_pair(n++, d); });
        return enumerated_data;
    }
} // namespace Sycl_Graph

#endif
