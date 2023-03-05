//include sycl
#include <CL/sycl.hpp>

int main()
{
    //get global and local memory size on GPU in Mb
    auto global_mem_size = sycl::device().get_info<sycl::info::device::global_mem_size>() / 1024 / 1024;
    auto local_mem_size = sycl::device().get_info<sycl::info::device::local_mem_size>() / 1024;
    std::cout << "Global memory size: " << global_mem_size << " Mb" << std::endl;
    std::cout << "Local memory size: " << local_mem_size << " Kb" << std::endl;

    
    return 0;

}
