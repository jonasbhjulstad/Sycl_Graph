
# set(SYCL_GRAPH_PRECOMPILE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../build/pcm")

file(MAKE_DIRECTORY ${SYCL_GRAPH_PRECOMPILE_DIR})


file(GLOB_RECURSE SYCL_GRAPH_MODULES "${PROJECT_SOURCE_DIR}/include/*.cxx")

# add_custom_target(PCM_Operations COMMAND "${CMAKE_CXX_COMPILER} -std=c++2a -c $ -Xclang -emit-module-interface -o ${SYCL_GRAPH_PRECOMPILE_DIR}/Operations.pcm")
set(SYCL_GRAPH_PCM_TARGETS)


foreach(module ${SYCL_GRAPH_MODULES})
  #get basename
  get_filename_component(basename ${module} NAME_WE)
  #remove file extension
  string(REGEX REPLACE ".cxx" "" basename_no_ext ${basename})
  #if basename_no_ext is a target, prepend Sycl
  if(TARGET PCM_${basename_no_ext})
    set(basename_no_ext "Sycl_${basename_no_ext}")
  endif()
  add_custom_target(PCM_${basename_no_ext}
  DEPENDS ${module})
  add_custom_command(
    TARGET PCM_${basename_no_ext}
    COMMAND ${CMAKE_CXX_COMPILER} -std=c++2a -c ${module} -fsycl -I ${PROJECT_SOURCE_DIR}/include -Xclang -emit-module-interface -o ${SYCL_GRAPH_PRECOMPILE_DIR}/${basename_no_ext}.pcm
    COMMENT "Precompiling ${basename_no_ext} to ${SYCL_GRAPH_PRECOMPILE_DIR}/${basename_no_ext}.pcm"
    PRE_BUILD
  )

  list(APPEND SYCL_GRAPH_PCM_TARGETS PCM_${basename_no_ext})
endforeach()