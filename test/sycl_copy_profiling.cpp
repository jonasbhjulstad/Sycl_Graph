#include <CL/sycl.hpp>


int main()
{
    //enable profiling
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::enable_profiling{});
    constexpr size_t N_elements = 100000000;
    //test sycl copy profiling
    sycl::buffer<int, 1> buf(N_elements);
    auto e = q.submit([&](sycl::handler &h)
    {
        auto acc = buf.get_access<sycl::access::mode::write>(h);
        h.parallel_for(sycl::range<1>{N_elements}, [=](sycl::id<1> it)
        {
            acc[it] = it[0];
        });
    });
    q.wait();
    // std::cout << "Time to copy: " << e.get_profiling_info<sycl::info::event_profiling::command_end>() - e.get_profiling_info<sycl::info::event_profiling::command_start>() << std::endl;
    //print in microseconds
    std::cout << "Time to copy: " << (e.get_profiling_info<sycl::info::event_profiling::command_end>() - e.get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000 << " us" << std::endl;
    std::cout << "ms: " << (e.get_profiling_info<sycl::info::event_profiling::command_end>() - e.get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000 / 1000 << " ms" << std::endl;
    //get opencl context
    auto cl_context = q.get_context();
    return 0;
}