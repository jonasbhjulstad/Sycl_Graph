cmake_minimum_required(VERSION 3.14...3.22)

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
set(SYCL_GRAPH_CUDA_FLAGS -fsycl-targets=nvptx64-nvidia-cuda)
option(SYCL_GRAPH_USE_CUDA "Enable CUDA support" ON)
include(cmake/external_packages.cmake)

set(SYCL_GRAPH_CLANG_BINARY_DIR "${IntelDPCPP_DIR}/../bin-llvm/clang++" CACHE STRING "Path to OneAPI clang/clang++ directory")

include(cmake/custom_executable.cmake)
include(cmake/ccache.cmake)
# ---- Add source files ----
add_compile_options(-Wdeprecated-declarations)
# Note: globbing sources is considered bad practice as CMake's generators may not detect new files
# automatically. Keep that in mind when changing files, or explicitly mention them here.
file(GLOB_RECURSE SYCL_GRAPH_HEADER_FILES "${PROJECT_SOURCE_DIR}/include/*.hpp")

# ---- Create library ----
add_library(${PROJECT_NAME} INTERFACE ${SYCL_GRAPH_HEADER_FILES})
target_include_directories(${PROJECT_NAME} INTERFACE "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>")
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 20)
target_link_libraries(${PROJECT_NAME} INTERFACE Tracy::TracyClient)
target_link_libraries(${PROJECT_NAME} INTERFACE cppitertools::cppitertools oneDPL)
target_compile_options(${PROJECT_NAME} INTERFACE ${DEFAULT_WARNING_FLAGS})

# Note: for header-only libraries change all PUBLIC flags to INTERFACE and create an interface
# target: add_library(${PROJECT_NAME} INTERFACE)

# being a cross-platform target, we enforce standards conformance on MSVC
target_compile_options(${PROJECT_NAME} INTERFACE "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/permissive->")

target_include_directories(
  ${PROJECT_NAME} INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                         $<INSTALL_INTERFACE:include/${PROJECT_NAME}-${PROJECT_VERSION}>
)

set(SYCL_GRAPH_DOXYGEN_IN ${PROJECT_SOURCE_DIR}/docs_doxygen/Doxyfile.in)
set(SYCL_GRAPH_DOXYGEN_OUT ${PROJECT_BINARY_DIR}/Doxyfile.out)
add_subdirectory(test)

# ---- Create an installable target ----
# this allows users to install and find the library via `find_package()`.

# the location where the project's version header will be placed should match the project's regular
# header paths
string(TOLOWER ${PROJECT_NAME}/version.h VERSION_HEADER_LOCATION)

packageProject(
  NAME ${PROJECT_NAME}
  VERSION ${PROJECT_VERSION}
  NAMESPACE ${PROJECT_NAME}
  BINARY_DIR ${PROJECT_BINARY_DIR}
  INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include
  INCLUDE_DESTINATION include/${PROJECT_NAME}-${PROJECT_VERSION}
  VERSION_HEADER "${VERSION_HEADER_LOCATION}"
  COMPATIBILITY SameMajorVersion
  DEPENDENCIES ""
)


