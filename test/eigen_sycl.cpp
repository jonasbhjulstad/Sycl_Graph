#include <Eigen/Dense>
#include <CL/sycl.hpp>
#include <iostream>

int main()
{
    typedef Eigen::Vector<float, 3> Vec;
    typedef Eigen::Matrix<float, 3, 3> Mat;
    Mat m;

    Vec v;

    sycl::queue q(sycl::gpu_selector_v);

    sycl::buffer<Mat, 1> m_buf(&m, sycl::range<1>(1));
    sycl::buffer<Vec, 1> v_buf(&v, sycl::range<1>(1));

    q.submit([&](sycl::handler& cgh) {
        auto m_acc = m_buf.get_access<sycl::access::mode::read_write>(cgh);
        auto v_acc = v_buf.get_access<sycl::access::mode::read_write>(cgh);
        cgh.parallel_for<class main>(sycl::range<1>(1), [=](sycl::id<1> i) {
            m_acc[i] = Mat::Identity();
            v_acc[i] = Vec::Ones();
        });
    });

    q.wait();

    //access buffer
    auto m_acc = m_buf.get_access<sycl::access::mode::read>();
    auto v_acc = v_buf.get_access<sycl::access::mode::read>();

    std::cout << m_acc[0] << std::endl;
    std::cout << v_acc[0] << std::endl;

    return 0;

}