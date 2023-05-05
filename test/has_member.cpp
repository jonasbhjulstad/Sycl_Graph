#include <concepts>
#include <type_traits>



template <typename T> 
concept test = requires(T t){typename T::Foo;};


template <typename T>
constexpr bool has_Foo(T& t)
{
    return test<T>;
}

struct FooHolder
{
    typedef int Foo;
};

struct FooLess
{
    typedef int Bar;
};

int main()
{
    FooHolder f;
    static_assert(has_Foo(f) == true);
    FooLess fl;
    static_assert(has_Foo(fl) == false);
}