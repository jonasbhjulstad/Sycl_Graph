#include <iostream>
#include <utility>
#include <metal.hpp>
#include "FooHeader.hpp"


template <typename ... Ts>
struct FooInst : metal::invoke<metal::lambda<Foo>, metal::list<Ts ...>> {};

template struct FooInst<float, double>;

template struct Foo<float>;
template struct Foo<double>;

// void instantiate()
// {
//     metal::invoke<metal::lambda<Foo>, DTypes> Inst;
// }
