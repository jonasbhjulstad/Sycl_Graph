#include <vector>
#include <CL/sycl.hpp>
#include <type_traits>

struct Foo;
struct Foo
{
    //declare as sycl device copyable
    Foo() = default;
    int a;

};

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

    //create sycl queue
    sycl::queue q(sycl::gpu_selector_v);
    //print selected device

    //create sycl queue
    // sycl::queue q(sycl::gpu_selector_v);
    //print selected device
    std::cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << "\n";
    //create a Foo
    std::vector<Foo> fooVec(1);

    //create sycl buffer for a
    sycl::buffer<Foo, 1> b(fooVec.data(), 100);
    q.wait();
    q.submit([&](sycl::handler& cgh){
        auto acc = b.get_access<sycl::access::mode::read_write>(cgh);
        cgh.single_task<class foo>([=](){
            acc[0].a = 1;
        });
    }).wait();

    return 0;
}
