#include <Sycl_Graph/Graph/Sycl/Graph.hpp>
#include <Sycl_Graph/instantiation.hpp>
#include <metal.hpp>
// template <typename T>
// struct Foo{};
// // instantiate all distributions
// template <template <typename> typename Target_Class, typename First, typename... Types>
// struct Instantiator_1
// {
//     Instantiator_1()
//     {
//         Instantiator_1<Target_Class, Types...>();
//         Target_Class<First> Inst;
//     }
// };

// template <template <typename> typename Target_Class, typename Last>
// struct Instantiator_1<Target_Class, Last>
// {
//     Instantiator_1()
//     {
//         Target_Class<Last> Inst;
//     }
// };
// template struct Instantiator_1<Edge, float, double, int>;



namespace Sycl_Graph
{


template struct
