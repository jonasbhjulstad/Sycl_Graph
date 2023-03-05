#ifndef SYCL_GRAPH_ALGORITHMS_HPP
#define SYCL_GRAPH_ALGORITHMS_HPP
#include <CL/sycl.hpp>
#include <Sycl_Graph/Graph/Graph_Types.hpp>
namespace Sycl_Graph::Sycl::algorithms
{
  void bitonic_sort(auto &acc, sycl::handler &h)
  {
    // parallel for
    h.parallel_for<class bitonic_sort>(sycl::range<1>(acc.size()), [=](sycl::id<1> index)
                                       {
      //get the index of the element to sort
      int i = index[0];

      //bitonic sort
      for (int k = 2; k <= acc.size(); k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
          int ixj = i ^ j;          
          if (ixj > i) {
            if ((i & k) == 0 && acc[i] > acc[ixj]) {
              std::swap(acc[i], acc[ixj]);
            }
            if ((i & k) != 0 && acc[i] < acc[ixj]) {
              std::swap(acc[i], acc[ixj]);
            }
          }
        }
      } });
  }

  void bitonic_sort(auto &acc, sycl::handler &h, auto condition)
  {
    // parallel for
    h.parallel_for<class bitonic_sort>(sycl::range<1>(acc.size()), [=](sycl::id<1> index)
                                       {
      //get the index of the element to sort
      int i = index[0];

      //bitonic sort
      for (int k = 2; k <= acc.size(); k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
          int ixj = i ^ j;          
          if (ixj > i) {
            if ((i & k) == 0 && condition(acc[i], acc[ixj])) {
                std::swap(acc[i], acc[ixj]);
            }
            if ((i & k) != 0 && condition(acc[ixj], acc[i]))
            {
                std::swap(acc[i], acc[ixj]);
            }
          }
        }
      } });

  }

}

#endif