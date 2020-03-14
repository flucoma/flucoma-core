
include_guard()

function(set_if_toplevel)
  set(prefix ARG)
  set(singleValues VAR SUPERBUILD TOPLEVEL)
  set(noValues "")
  set(multiValues "")  
  include(CMakeParseArguments)
  cmake_parse_arguments(${prefix}
                       "${noValues}"
                       "${singleValues}"
                       "${multiValues}"
                        ${ARGN})                    
  if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR}) #toplevel project
    set(${ARG_VAR} ${ARG_TOPLEVEL} PARENT_SCOPE)
  else()
    set(${ARG_VAR} ${ARG_SUPERBUILD} PARENT_SCOPE)    
  endif()                             
endfunction()
