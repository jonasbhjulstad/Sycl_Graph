include(${PROJECT_SOURCE_DIR}/cmake/CPM.cmake)

# CPMFindPackage(
#     NAME Static_RNG
#     GITHUB_REPOSITORY jonasbhjulstad/Static_RNG
#     GIT_TAG master
#     OPTIONS
#     STATIC_RNG_ENABLE_SYCL OFF
#     BUILD_PYTHON_BINDERS OFF
#     BUILD_DOCS OFF
# )
find_package(Static_RNG REQUIRED)

CPMFindPackage(
    NAME tinymt
    GITHUB_REPOSITORY tueda/tinymt-cpp
    GIT_TAG master
    OPTIONS
    "BUILD_TESTING OFF"
)

find_package(oneDPL REQUIRED)
find_package(pybind11 CONFIG HINTS ${PYTHON_ENV_CMAKE_MODULE_DIR})
# CPMFindPackage(
#     NAME pybind11
#     GITHUB_REPOSITORY pybind/pybind11
#     GIT_TAG master
# )
# include(FindPkgConfig)
# pkg_check_modules(graph_tool REQUIRED graph-tool-py3.9)

# set(graph_tool_LIBRARY_DIR "/home/man/.conda/envs/py39/lib/python3.9/site-packages/graph_tool")

# set(graph_tool_libraries graph_tool_core)

# message(STATUS "graph-tool-py3.9_INCLUDE_DIRS: ${graph_tool_LIBRARIES}")
find_package(Boost 1.78 REQUIRED HINTS ${PYTHON_ENV_CMAKE_MODULE_DIR})
# find_package(Static_RNG REQUIRED)

set(cppitertools_INSTALL_CMAKE_DIR share)
CPMFindPackage(
    NAME cppitertools
    GITHUB_REPOSITORY ryanhaining/cppitertools
    GIT_TAG master
    OPTIONS
    "cppitertools_INSTALL_CMAKE_DIR share"
)

# CPMAddPackage(
#     NAME oneDPL
#     GITHUB_REPOSITORY oneapi-src/oneDPL
#     GIT_TAG master
#     OPTIONS
#     "-DCMAKE_BUILD_TYPE=Release"

# )
find_package(TBB REQUIRED)
include(FindThreads)
# CPMFindPackage(
#     NAME Tracy
#     GITHUB_REPOSITORY wolfpld/tracy
#     GIT_TAG master
# )

find_package(TBB REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

CPMAddPackage("gh:TheLartians/PackageProject.cmake@1.6.0")
