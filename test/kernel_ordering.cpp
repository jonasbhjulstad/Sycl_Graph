#include <CL/sycl.hpp>

int main()
{
    sycl::queue q(sycl::gpu_selector_v);

    sycl::buffer<int, 1> buf{ sycl::range<1>{ 1 } };

    auto event = q.submit([&](sycl::handler& cgh) {
        auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);
        cgh.single_task<class my_kernel>([=]() {
            acc[0] = 42;
        });
    });

    q.submit([&](sycl::handler& cgh) {
        auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);
        cgh.depends_on(event);
        cgh.single_task<class my_kernel2>([=]() {
            acc[0] = 43;
        });
    });
    std::cout << buf.get_access<sycl::access::mode::read>()[0] << std::endl;
    return 0;
}