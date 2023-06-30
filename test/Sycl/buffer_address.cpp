#include <CL/sycl.hpp>

int main()
{
    sycl::buffer<int> iBuf(sycl::range<1>(100));

    auto cl_mem_buf = iBuf.get_buffer();
}
