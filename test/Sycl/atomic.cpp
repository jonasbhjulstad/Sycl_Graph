#include <vector>
#include <CL/sycl.hpp>
#include <type_traits>


int main()
{
    //get all gpu devices
    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);

    auto version = devices[1].get_info<cl::sycl::info::device::driver_version>();
    std::cout << version << std::endl;


    //print them
for(auto& device : devices)
    {
        std::cout << device.get_info<sycl::info::device::name>() << "\n";
    }
    //get max work group size
    auto maxWorkGroupSize = devices[1].get_info<sycl::info::device::max_work_group_size>();
    //create sycl queue
    sycl::queue q(sycl::gpu_selector_v);
    //print selected device

    //create sycl queue
    // sycl::queue q(sycl::gpu_selector_v);
    //print selected device
    std::cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << "\n";
    //create a Foo
    std::vector<uint32_t> fooVec(1);

    //create sycl buffer for a
    sycl::buffer<uint32_t, 1> b(fooVec.data(), 100);
    q.wait();
    q.submit([&](sycl::handler& cgh){
        auto acc = b.get_access<sycl::access::mode::atomic>(cgh);
        cgh.parallel_for(maxWorkGroupSize, [=](sycl::id<1> id)
        {
            sycl::atomic_fetch_add<uint32_t>(acc[0], 1);
        });
    }).wait();

    //read back
    auto acc = b.get_access<sycl::access::mode::read>();
    std::cout << acc[0] << std::endl;

    return 0;
}
