#ifndef SYCL_GRAPH_UTILS_RNG_GENERATION_HPP
#define SYCL_GRAPH_UTILS_RNG_GENERATION_HPP
#include <random>
#include <Sycl_Graph/Utils/Buffer_Utils.hpp>
namespace Sycl_Graph
{
    auto generate_seeds(auto N, auto seed)
    {
        std::mt19937 gen(seed);
        std::vector<uint32_t> seeds(N);
        for(int i = 0; i < N; i++)
        {
            seeds[i] = gen();
        }
        return seeds;
    }

    template <typename RNG>
    auto generate_rngs(auto N, auto seed)
    {
        auto seeds = generate_seeds(N, seed);
        std::vector<RNG> rngs;
        rngs.reserve(N);
        std::transform(seeds.begin(), seeds.end(), std::back_inserter(rngs), [](auto seed)
        {
            return RNG(seed);
        });
        return rngs;
    }

}

namespace Sycl_Graph::USM
{
    template <typename RNG>
    auto generate_shared_usm_rngs(sycl::queue& q, auto N, auto seed)
    {
        auto rngs = generate_rngs<RNG>(N, seed);
        return USM::make_shared_usm<RNG>(q, rngs);
    }

    template <typename RNG>
    auto generate_usm_rngs(sycl::queue& q, auto N, auto seed, sycl::event& event)
    {
        auto rngs = generate_rngs<RNG>(N, seed);
        auto p_rngs = sycl::malloc_device<RNG>(N, q);
        event = q.submit([&](sycl::handler& h)
        {
            h.copy(rngs.data(), p_rngs, N);
        });
        return p_rngs;
    }
}

#endif
