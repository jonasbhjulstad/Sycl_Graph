#include <Sycl_Graph/Buffer/Sycl/Buffer_Routines.hpp>

std::vector<int> initial_i = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
std::vector<float> initial_f = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
int main()
{

    sycl::queue q(sycl::gpu_selector_v);

    sycl::buffer<int, 1> ibuf(initial_i.data(), sycl::range<1>(initial_i.size()));
    sycl::buffer<float, 1> fbuf(initial_f.data(), sycl::range<1>(initial_f.size()));
    //10 + initial_i
    std::vector<int> ivector = {10, 11, 12, 13, 2, 15, 16, 17, 18, 6};
    std::vector<float> fvector = {10.0f, 11.0f, 12.0f, 13.0f, 2.0f, 15.0f, 16.0f, 17.0f, 18.0f, 6.0f};

    q.wait();

    Sycl_Graph::buffer_print(ibuf, q, "ibuf");
    Sycl_Graph::buffer_print(fbuf, q, "fbuf");

    const auto vectup = std::make_tuple(ivector, fvector);

    auto buftup = std::make_tuple(ibuf, fbuf);

    Sycl_Graph::buffer_assign_add<int, uint32_t, int, float>(buftup, q, vectup);

    std::cout << "New buffer size: " << std::get<0>(buftup).size() << std::endl;
    Sycl_Graph::buffer_print(std::get<0>(buftup), q, "ibuf");
    Sycl_Graph::buffer_print(std::get<1>(buftup), q, "fbuf");

    return 0;
}