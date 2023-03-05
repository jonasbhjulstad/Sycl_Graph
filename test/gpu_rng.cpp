#include "tinymt/tinymt.h"
#include <CL/sycl.hpp>
#include <Static_RNG/distributions.hpp>
#include <iostream>

int main()
{
    size_t N_rngs = 4;
    sycl::buffer<Static_RNG::default_rng, 1> rng_buf(N_rngs);
    std::vector<unsigned int> seeds(N_rngs);
    for (size_t i = 0; i < N_rngs; ++i)
    {
        seeds[i] = i;
    }
    
    sycl::buffer<unsigned int, 1> seed_buf(seeds);
    sycl::buffer<unsigned int, 1> result_buf(seeds);
    sycl::queue q(sycl::gpu_selector{});
    q.submit([&](sycl::handler &cgh) {
        auto rng_acc = rng_buf.get_access<sycl::access::mode::write>(cgh);
        auto seed_acc = seed_buf.get_access<sycl::access::mode::read>(cgh);
        auto result_acc = result_buf.get_access<sycl::access::mode::write>(cgh);
        cgh.single_task<class tinymt32_kernel>([=]() {
            for (size_t i = 0; i < N_rngs; ++i)
            {
                rng_acc[i].seed(seed_acc[i]);
                result_acc[i] = rng_acc[i]();
            }
        });
    });

    //print result
    auto acc = result_buf.get_access<sycl::access::mode::read>();
    for (size_t i = 0; i < N_rngs; ++i)
    {
        std::cout << acc[i] << std::endl;
    }
    return 0;
}