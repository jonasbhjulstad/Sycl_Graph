#include <tuple>
#include <iostream>

int main()
{
    //demonstrate tuple_cat
    std::tuple<int, char, float> t1(1, 'a', 3.14);
    std::tuple<double, char> t2(2.71, 'b');
    auto t3 = std::tuple_cat(t1, t2);

    std::tuple<std::tuple<int>> iT;
    std::tuple<std::tuple<float>> fT;



    auto ifT = std::tuple_cat(std::make_tuple(iT), fT);

    //print ifT

    return 0;
}
