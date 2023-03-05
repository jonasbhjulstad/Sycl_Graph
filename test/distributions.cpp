#include "tinymt/tinymt.h"
#include <CL/sycl.hpp>
#include <Static_RNG/distributions.hpp>
#include <iostream>
#include <random>

int main()
{
    size_t N_rngs = 4;
    sycl::buffer<Static_RNG::default_rng, 1> rng_buf(N_rngs);
    std::vector<uint_fast32_t> seeds(N_rngs);
    for (size_t i = 0; i < N_rngs; ++i)
    {
        seeds[i] = 1000*i;
    }
    
    tinymt::tinymt32 mt(928);
    Static_RNG::uniform_real_distribution<float> dist(0, 1);
    for (size_t i = 0; i < N_rngs; ++i)
    {
        std::cout << dist(mt) << std::endl;
    }

    Static_RNG::binomial_distribution<> dist2(10, 0.5);
    for (size_t i = 0; i < N_rngs; ++i)
    {
        std::cout << dist2(mt) << std::endl;
    }

    sycl::buffer<uint_fast32_t, 1> seed_buf(seeds);
    sycl::buffer<float, 1> result_buf((sycl::range<1>(N_rngs)));
    sycl::buffer<Static_RNG::binomial_distribution<> , 1> dist_buf(N_rngs);
    sycl::queue q(sycl::gpu_selector{});
    q.submit([&](sycl::handler &cgh) {
        auto rng_acc = rng_buf.get_access<sycl::access::mode::read_write>(cgh);
        auto seed_acc = seed_buf.get_access<sycl::access::mode::read>(cgh);
        auto result_acc = result_buf.template get_access<sycl::access::mode::write>(cgh);
        auto dist_acc = dist_buf.get_access<sycl::access::mode::read_write>(cgh);
        cgh.single_task<class tinymt32_kernel>([=]() {
            for (size_t i = 0; i < N_rngs; ++i)
            {
                dist_acc[i].set_trials(10);
                dist_acc[i].set_probability(0.5);

                rng_acc[i].seed(seed_acc[i]);
                result_acc[i] = dist_acc[i](rng_acc[i]);
            }
        });
    });
    q.wait();

    //print result
    auto acc = result_buf.get_access<sycl::access::mode::read>();
    for (size_t i = 0; i < N_rngs; ++i)
    {
        std::cout << acc[i] << std::endl;
    }
    return 0;
}