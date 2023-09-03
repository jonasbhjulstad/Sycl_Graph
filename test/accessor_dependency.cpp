#include <CL/sycl.hpp>




int main()
{
    cl::sycl::property_list propList{cl::sycl::property::queue::enable_profiling()};

    auto kernel = [](sycl::buffer<int, 2>& buf, auto row_offset){
        return [&](sycl::handler& h)
    {
        auto acc = sycl::accessor<int, 2, sycl::access::mode::read_write, sycl::access::target::global_buffer>(buf, h, sycl::range<2>(1, 1), sycl::id<2>(row_offset, 0));
    };
    };

    sycl::buffer<int, 2> buf(sycl::range<2>(, 1));
    sycl::queue q(sycl::gpu_selector_v, propList);
    std::vector<sycl::event> events(N);
    for(auto row_offset = 0; row_offset < N; row_offset++)
    {
        events[row_offset] = q.submit(kernel(buf, row_offset));
    }
    std::for_each(row_offset.begin(), row_offset.end(), [](auto e){e.wait();});
    return 0;
}
