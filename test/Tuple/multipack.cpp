void foo(auto& ... args_0, const auto& ... args_1) {
    std::cout << sizeof...(args_0) << std::endl;
    std::cout << sizeof...(args_1) << std::endl;
}

int main()
{

}
