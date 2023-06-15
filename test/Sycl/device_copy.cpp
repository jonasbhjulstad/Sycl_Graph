#include <vector>
#include <CL/sycl.hpp>
#include <type_traits>

struct Foo;
struct Foo
{
    //declare as sycl device copyable
    Foo() = default;
    Foo(Foo& a): d(a.d){}
    std::vector<int> d;

};
template <>
inline constexpr bool sycl::is_device_copyable_v<Foo> = true;

int main()
{
    //get all gpu devices
    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);

    //print them
for(auto& device : devices)
    {
        std::cout << device.get_info<sycl::info::device::name>() << "\n";
    }

    //create sycl queue
    sycl::queue q(devices[1]);
    //print selected device

    //create sycl queue
    // sycl::queue q(sycl::gpu_selector_v);
    //print selected device
std::cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << "\n";
    //create a Foo
    Foo a;
    //create sycl buffer for a
    sycl::buffer<Foo, 1> b(&a, sycl::range<1>(1));
    q.submit([&](sycl::handler& cgh){
        auto acc = b.get_access<sycl::access::mode::read_write>(cgh);
        cgh.single_task<class foo>([=](){
            acc[0].d[0] = 1;
        });
    });
}
