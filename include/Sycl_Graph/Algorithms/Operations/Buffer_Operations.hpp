#ifndef SYCL_GRAPH_ALGORITHMS_PROPERTIES_BUFFER_OPERATIONS_HPP
#define SYCL_GRAPH_ALGORITHMS_PROPERTIES_BUFFER_OPERATIONS_HPP
#include <Sycl_Graph/Algorithms/Operations/Operation_Types.hpp>
namespace Sycl_Graph::Sycl {

  template <Operation_type Op>
  sycl::event buffer_transform(sycl::queue& q, Op& operation,
                               sycl::buffer<typename Op::Source_t>& source_buf,
                               sycl::buffer<typename Op::Target_t>& target_buf,
                               sycl::event dep_event = {}) {
    return q.submit([&](sycl::handler& h) {
      h.depends_on(dep_event);
      auto source_acc = source_buf.template get_access<Op::source_access_mode>(h);
      auto target_acc = target_buf.template get_access<sycl::access_mode::write>(h);
      operation(source_acc, target_acc, h);
    });
  }

  template <Operation_type Op>
  sycl::event buffer_inplace_modification(sycl::queue& q, Op& operation,
  sycl::buffer<typename Op::Inplace_t>& inplace_buf,sycl::event dep_event = {})
  {
    return q.submit([&](sycl::handler& h) {
      h.depends_on(dep_event);
      auto inplace_acc = inplace_buf.template get_access<Op::inplace_access_mode>(h);
      operation(inplace_acc, h);
    });
  }
  
}
#endif