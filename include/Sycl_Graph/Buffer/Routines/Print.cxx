module;
#include <Sycl_Graph/Common/common.hpp>
export module Sycl.Buffer.Print;

    export template <typename T>
    void buffer_print(sycl::buffer<T, 1> &buf, sycl::queue &q,
                             const std::string &name = "") {
  if (buf.size() == 0) {
    return;
  }

  // if T is integral or floating point
  if constexpr (std::is_integral_v<T> || std::is_floating_point_v<T>) {
    std::string type_name = typeid(T).name();
    std::cout << ((name == "") ? type_name : name) << ": ";
    q.submit([&](sycl::handler &h) {
       auto acc = buf.template get_access<sycl::access::mode::read>(h);
       sycl::stream out(1024, 256, h);
       h.single_task([=]() {
         for (int i = 0; i < acc.size(); i++) {
           out << acc[i] << ", ";
         }
       });
     }).wait();
    std::cout << std::endl;
  }
  return;
}

template <typename... Ts>
export void buffer_print(std::tuple<sycl::buffer<Ts, 1>...> &bufs, sycl::queue &q) {
  std::apply([&](auto &...buf) { (buffer_print(buf, q), ...); }, bufs);
}