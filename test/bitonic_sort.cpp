#include <vector>
#include <CL/sycl.hpp>
#include <iostream>
#include <algorithm>
// include for assert
#include <cassert>

//create sycl kernel sorting example
int main()
{
  //create a vector of 8 elements
  std::vector<int> v = { 3, 1, 4, 1, 5, 9, 2, 6 };


  //create a queue
  sycl::queue q;
  {
  //create a buffer of 8 elements
  sycl::buffer<int, 1> buf(v.data(), sycl::range<1>(v.size()));

  //create a command group to issue command to the queue
  q.submit([&](sycl::handler &cgh) {
    //create an accessor
    auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);

    //parallel for
    cgh.parallel_for<class bitonic_sort>(sycl::range<1>(v.size()), [=](sycl::id<1> index) {
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
      }
    });
  });
  }

  //wait for the queue to finish
  q.wait();

//print v
  for (auto i : v) {
    std::cout << i << " ";
  }

  return 0;

}

