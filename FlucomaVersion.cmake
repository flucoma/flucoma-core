# Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
# Copyright University of Huddersfield.
# Licensed under the BSD-3 License.
# See license.md file in the project root for full license information.
# This project has received funding from the European Research Council (ERC)
# under the European Union’s Horizon 2020 research and innovation programme
# (grant agreement No 725899).

include_guard() 

cmake_minimum_required (VERSION 3.11)

find_package(Git REQUIRED)

set(flucoma_VERSION_MAJOR 1)
set(flucoma_VERSION_MINOR 0)
set(flucoma_VERSION_PATCH 8)
set(flucoma_VERSION_SUFFIX "")

function(make_flucoma_version_string output_variable)
  set(result "${flucoma_VERSION_MAJOR}.${flucoma_VERSION_MINOR}.${flucoma_VERSION_PATCH}") 
  if(flucoma_VERSION_SUFFIX)
    string(APPEND result "-${flucoma_VERSION_SUFFIX}")
  endif()  
  set(${output_variable} ${result} PARENT_SCOPE)
endfunction()

function(make_flucoma_version_string_with_sha output_variable)
  
  make_flucoma_version_string(${output_variable})
  
  if(NOT flucoma_CORESHA_ERR AND NOT flucoma_VERSIONSHA_ERR)
    string(APPEND 
      ${output_variable}
      "+sha.${flucoma_VERSION_SHA}.core.sha.${flucoma_CORE_SHA}"
    )
  endif()   
  set(${output_variable} ${${output_variable}} PARENT_SCOPE)
endfunction()

macro(getsha working_dir output_variable result_variable)
  execute_process( 
    COMMAND ${GIT_EXECUTABLE} log -1 --format=%h
    WORKING_DIRECTORY ${${working_dir}}
    OUTPUT_VARIABLE ${output_variable}
    RESULT_VARIABLE ${result_variable}
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE  
  )
endmacro() 

getSha(CMAKE_CURRENT_LIST_DIR flucoma_CORE_SHA flucoma_CORESHA_ERR)
getSha(CMAKE_SOURCE_DIR flucoma_VERSION_SHA flucoma_VERSIONSHA_ERR)

make_flucoma_version_string(flucoma_VERSION_STRING)

make_flucoma_version_string_with_sha(flucoma_VERSION_STRING_SHA)

file(WRITE 
  "${CMAKE_CURRENT_SOURCE_DIR}/flucoma.version.rc" "${flucoma_VERSION_STRING}"
)

configure_file("${CMAKE_CURRENT_LIST_DIR}/script/FluidVersion.cpp.in" "${CMAKE_CURRENT_LIST_DIR}/FluidVersion.cpp" @ONLY)

add_library(flucoma_VERSION_LIB STATIC 
  "${CMAKE_CURRENT_LIST_DIR}/FluidVersion.cpp"   
)

target_include_directories(flucoma_VERSION_LIB PRIVATE  
  "${CMAKE_CURRENT_LIST_DIR}/include"
)

set_target_properties(flucoma_VERSION_LIB PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

set_property(GLOBAL PROPERTY FLUCOMA_VERSION ${flucoma_VERSION_STRING})
set_property(GLOBAL PROPERTY FLUCOMA_VERSION_TERSE ${flucoma_VERSION_MAJOR}.${flucoma_VERSION_MINOR}.${flucoma_VERSION_PATCH})
set_property(GLOBAL PROPERTY FLUCOMA_VERSION_SHA ${flucoma_VERSION_STRING_SHA})
set_property(GLOBAL PROPERTY FLUCOMA_VERSION_MAJOR ${flucoma_VERSION_MAJOR})
set_property(GLOBAL PROPERTY FLUCOMA_VERSION_MINOR ${flucoma_VERSION_MINOR})
set_property(GLOBAL PROPERTY FLUCOMA_VERSION_PATCH ${flucoma_VERSION_PATCH})
set_property(GLOBAL PROPERTY FLUCOMA_VERSION_SUFFIX ${flucoma_VERSION_SUFFIX})
