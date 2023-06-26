struct Foo
{
    using FooType = int;
};

struct Bar: Foo
{

};


int main()
{
    typename Bar::FooType a;
    return 0;
}
