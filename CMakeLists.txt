cmake_minimum_required(VERSION 3.26)

# ---- Project ----

# Note: update this to your new project's name and version
project(
  Sycl_Graph
  VERSION 1.0
  LANGUAGES CXX
)

# ---- Include guards ----
if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(
    FATAL_ERROR
      "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there."
  )
endif()
set(DPCPP_CLANG_COMPILER /opt/intel/oneapi/compiler/2023.0.0/linux/bin-llvm/clang++)

string(CONCAT CMAKE_EXPERIMENTAL_CXX_SCANDEP_SOURCE
  "${DPCPP_CLANG_COMPILER} <DEFINES> <INCLUDES> <FLAGS> <SOURCE>"
  " -MT <DYNDEP_FILE> -MD -MF <DEP_FILE>"
  " ${flags_to_scan_deps}")# -fdep-file=<DYNDEP_FILE> -fdep-output=<OBJECT>")
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON) 
# set(CMAKE_CXX_EXTENSIONS OFF)
# set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API 2182bf5c-ef0d-489a-91da-49dbc3090d2a)
# set(CMAKE_EXPERIMENTAL_CXX_MODULE_DYNDEP 1)
set(SYCL_GRAPH_CUDA_FLAGS -fsycl-targets=nvptx64-nvidia-cuda)
set(SYCL_GRAPH_PRECOMPILE_DIR "${PROJECT_SOURCE_DIR}/build/pcm")
option(SYCL_GRAPH_USE_CUDA "Enable CUDA support" OFF)
include(cmake/CPM.cmake)
include(cmake/external_packages.cmake)
include(cmake/Module_Precompilation.cmake)
set(SYCL_GRAPH_CLANG_BINARY_DIR "${IntelDPCPP_DIR}/../bin-llvm/clang++" CACHE STRING "Path to OneAPI clang/clang++ directory")
set(CMAKE_COLOR_MAKEFILE ON)
include(cmake/custom_executable.cmake)
# include(cmake/ccache.cmake)
list(APPEND CMAKE_MODULE_PATH "/home/man/Downloads/cmake-checks-cache/")
# ---- Add source files ----
add_compile_options(-Wdeprecated-declarations)
# Note: globbing sources is considered bad practice as CMake's generators may not detect new files
# automatically. Keep that in mind when changing files, or explicitly mention them here.
file(GLOB_RECURSE SYCL_GRAPH_HEADER_FILES "${PROJECT_SOURCE_DIR}/include/*.hpp")
# ---- Create library ----
file(GLOB_RECURSE SYCL_GRAPH_INCLUDE_MODULES "${PROJECT_SOURCE_DIR}/include/*.ixx")
file(GLOB_RECURSE SYCL_GRAPH_OPERATION_MODULES "${PROJECT_SOURCE_DIR}/include/Sycl_Graph/Operations/*.ixx")
file(GLOB_RECURSE SYCL_GRAPH_GRAPH_MODULES "${PROJECT_SOURCE_DIR}/include/Sycl_Graph/Graph/*.ixx")
file(GLOB_RECURSE SYCL_GRAPH_BUFFER_MODULES "${PROJECT_SOURCE_DIR}/include/Sycl_Graph/Buffer/*.ixx")

# add_custom_target(SYCL_GRAPH_PCM
# COMMAND ${CMAKE_COMMAND} -P ${PROJECT_SOURCE_DIR}/cmake/Module_Precompilation.cmake)


add_subdirectory(include/Sycl_Graph)
# set(SYCL_GRAPH_DOXYGEN_IN ${PROJECT_SOURCE_DIR}/docs_doxygen/Doxyfile.in)
# set(SYCL_GRAPH_DOXYGEN_OUT ${PROJECT_BINARY_DIR}/Doxyfile.out)
add_subdirectory(test)
add_compile_options("-Xclang -fcolor-diagnostics")
add_compile_options("-v")
# add_compile_options("-fdiagnostics-format=gcc")
# ---- Create an installable target ----
# this allows users to install and find the library via `find_package()`.

# the location where the project's version header will be placed should match the project's regular
# header paths
# string(TOLOWER ${PROJECT_NAME}/version.h VERSION_HEADER_LOCATION)

# packageProject(
#   NAME ${PROJECT_NAME}
#   VERSION ${PROJECT_VERSION}
#   NAMESPACE ${PROJECT_NAME}
#   BINARY_DIR ${PROJECT_BINARY_DIR}
#   INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include
#   INCLUDE_DESTINATION include/${PROJECT_NAME}-${PROJECT_VERSION}
#   VERSION_HEADER "${VERSION_HEADER_LOCATION}"
#   COMPATIBILITY SameMajorVersion
#   DEPENDENCIES ""
# )


