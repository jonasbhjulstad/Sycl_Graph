#include <CL/sycl.hpp>

int main()
{
    sycl::buffer<int> buf(1000);

    std::cout << sizeof(buf) << std::endl;
}