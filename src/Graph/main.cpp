#include "FooHeader.hpp"
int main()
{
    Foo<float> foo;
    Foo<int> iFoo;
    iFoo.FooFun();
    foo.FooFun();
}
