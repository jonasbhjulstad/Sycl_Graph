include(cmake/CPM.cmake)

CPMFindPackage(
    NAME Static_RNG
    GITHUB_REPOSITORY jonasbhjulstad/Static_RNG
    GIT_TAG master
    OPTIONS
    STATIC_RNG_ENABLE_SYCL ON
    BUILD_PYTHON_BINDERS OFF
    BUILD_DOCS OFF
)

find_package(IntelDPCPP REQUIRED)


set(cppitertools_INSTALL_CMAKE_DIR share)
CPMFindPackage(
    NAME cppitertools
    GITHUB_REPOSITORY ryanhaining/cppitertools
    GIT_TAG master
    OPTIONS
    "cppitertools_INSTALL_CMAKE_DIR share"
)
include(FindThreads)

option(TRACY_ENABLE "" ON)
option (TRACY_ON_DEMAND "" ON)
CPMFindPackage(
    NAME Tracy
    GITHUB_REPOSITORY wolfpld/tracy
    GIT_TAG master
)

find_package(oneDPL REQUIRED)

# CPMFindPackage(NAME etl
#     GITHUB_REPOSITORY  ETLCPP/etl
#     GIT_TAG master    
# )

# CPMFindPackage(NAME Metal
#     GITHUB_REPOSITORY brunocodutra/metal
#     GIT_TAG master
# )

find_package(Doxygen QUIET)

# CPMFindPackage(
#     NAME Eigen3
#     GITHUB_REPOSITORY libigl/eigen
#     GIT_TAG master
#     OPTIONS
#     "QUIET ON"
# )
# CPMFindPackage(
#     NAME DataFrame
#     GITHUB_REPOSITORY hosseinmoein/DataFrame
#     GIT_TAG master
# )

CPMAddPackage("gh:TheLartians/PackageProject.cmake@1.6.0")


#boost graph library
find_package(Boost REQUIRED COMPONENTS graph)