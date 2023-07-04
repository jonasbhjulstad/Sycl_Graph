#ifndef SYCL_GRAPH_ALGORITHMS_OPERATIONS_DEFAULT_OPERATIONS_HPP
#define SYCL_GRAPH_ALGORITHMS_OPERATIONS_DEFAULT_OPERATIONS_HPP
#include <Sycl_Graph/Algorithms/Operations/Operation_Types.hpp>
namespace Sycl_Graph::Sycl {
  template <typename... Source_Ts> struct Buffer_Copy
      : public Operation_Base
        < Buffer_Copy<Source_Ts...>, Read_Accessors_t<Source_Ts...>,
                                   Write_Accessors_t<Source_Ts...>> {

    void invoke(const sycl::accessor<Source_Ts, 1, sycl::access_mode::read>& ... source_acc, sycl::accessor<Source_Ts, 1, sycl::access_mode::write>& ... target_acc, sycl::handler &h) {
      h.parallel_for([=](sycl::range<1> id) {
        (target_acc[id] = source_acc[id], ...);
      });
    }
    template <typename T, typename Graph_t> int get_buffer_size(const Graph_t &G) const {
      return OPERATION_SIZE_INHERITED;
    }
  };

}  // namespace Sycl_Graph::Sycl

#endif
