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
    //create sycl queue
    sycl::queue q;
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