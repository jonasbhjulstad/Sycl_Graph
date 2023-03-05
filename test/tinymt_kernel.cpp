#include <Static_RNG/distributions.hpp>
#include <CL/sycl.hpp>
#include <tinymt/tinymt.h>


int main()
{
    //test tinymt32 in kernel

    //create queue on gpu
    sycl::queue q(sycl::gpu_selector_v);

    sycl::buffer<float, 1> buf(sycl::range<1>(2));

    sycl::buffer<tinymt::tinymt32, 1> rng_buf(sycl::range<1>(1));
    sycl::buffer<Static_RNG::uniform_real_distribution<>, 1> dist_buf(sycl::range<1>(1));
    q.submit([&](sycl::handler &cgh) {
        auto acc = buf.get_access<sycl::access::mode::write>(cgh);
        auto rng_acc = rng_buf.get_access<sycl::access::mode::read_write>(cgh);
        auto dist_acc = dist_buf.get_access<sycl::access::mode::read_write>(cgh);
        cgh.single_task<class tinymt32_kernel>([=]() {
            // tinymt::tinymt32_init(&rng, 0);
            // dist_acc[0].set_trials(10);
            // dist_acc[0].set_probability(0.5);
            // rng_acc[0].seed(100);
            tinymt::tinymt32 mt;
            acc[0] = dist_acc[0](rng_acc[0]);
            acc[1] = dist_acc[0](rng_acc[0]);
            acc[0] = dist_acc[0](mt);
            acc[1] = dist_acc[1](mt);
            // acc[0] = tinymt::tinymt32_generate_uint32(&rng);
        });
    });
    q.wait();
    //print result
    auto acc = buf.get_access<sycl::access::mode::read>();
    std::cout << acc[0] << ", " << acc[1] << std::endl;
}