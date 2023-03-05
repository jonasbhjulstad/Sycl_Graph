#include <CL/sycl.hpp>
#include <iostream>

int main()
{
    //demonstrate the usage of kernel bundles
    //get context
    sycl::context ctx = sycl::context();
    sycl::kernel_bundle<sycl::bundle_state::input> kb = sycl::get_kernel_bundle<sycl::bundle_state::input>(ctx);
    std::cout << "kernel bundle size: " << kb.get_kernel_ids().size() << std::endl;
    return 0;

}