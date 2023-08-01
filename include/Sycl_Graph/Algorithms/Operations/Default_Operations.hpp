#ifndef SYCL_GRAPH_ALGORITHMS_OPERATIONS_DEFAULT_OPERATIONS_HPP
#define SYCL_GRAPH_ALGORITHMS_OPERATIONS_DEFAULT_OPERATIONS_HPP
#include <Sycl_Graph/Algorithms/Operations/Operation_Types.hpp>
namespace Sycl_Graph::Sycl {

  template <typename Source_t, typename Target_t> struct Copy_Operation
      : public Operation_Base<Copy_Operation<Source_t, Target_t>,
                              Accessor_t<Source_t, sycl::access_mode::read>,
                              Accessor_t<Target_t, sycl::access_mode::write>> {
    using Base_t = Operation_Base<Copy_Operation<Source_t, Target_t>,
                                  Accessor_t<Source_t, sycl::access_mode::read>,
                                  Accessor_t<Target_t, sycl::access_mode::write>>;
    void invoke(sycl::handler& h, const auto& source_acc, auto& target_acc) {
                                    h.parallel_for(target_acc.size(), [=](sycl::id<1> id)
                                    {
        target_acc[id] = source_acc[id];
                                    }
    }
  };

}  // namespace Sycl_Graph::Sycl

#endif
