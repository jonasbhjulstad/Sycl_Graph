cmake_minimum_required(VERSION 3.26)

project(Instantiation_Example)

add_library(FloatFoo STATIC FloatFoo.cpp)
add_library(IntFoo STATIC IntFoo.cpp)

add_executable(Instantiation_Example main.cpp)
target_link_libraries(Instantiation_Example PRIVATE FloatFoo)
