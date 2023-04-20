#include <Sycl_Graph/type_helpers.hpp>

std::tuple<double, int, float, bool> t{1.0, 2, 3.0f, true};

using namespace Sycl_Graph;
int main()
{
    static_assert(index_of_type<int, double, int, float, bool>() == 1);

}