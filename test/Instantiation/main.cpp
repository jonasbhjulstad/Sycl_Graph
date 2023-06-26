#include "FooHeader.hpp"
extern template struct Foo<float>;
extern template struct Foo<int>;


int main()
{
    Foo<float> foo;
    Foo<int> iFoo;
    iFoo.FooFun();
    foo.FooFun();
}
