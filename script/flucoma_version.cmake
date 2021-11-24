# Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
# Copyright 2017-2019 University of Huddersfield.
# Licensed under the BSD-3 License.
# See license.md file in the project root for full license information.
# This project has received funding from the European Research Council (ERC)
# under the European Union’s Horizon 2020 research and innovation programme
# (grant agreement No 725899).

include_guard()

find_package(Git REQUIRED)

macro(getsha workingDir varName)
  execute_process( 
    COMMAND ${GIT_EXECUTABLE} log -1 --format=%h
    WORKING_DIRECTORY ${${workingDir}}
    OUTPUT_VARIABLE ${varName}
    # ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE  
  )
endmacro() 

getSha(CMAKE_CURRENT_LIST_DIR FLUID_CORE_SHA)
getSha(CMAKE_CURRENT_SOURCE_DIR FLUID_VERSION_SHA)

execute_process( 
  COMMAND ${GIT_EXECUTABLE} describe --abbrev=0 --always
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  RESULT_VARIABLE result
  OUTPUT_VARIABLE FLUID_VERSION_TAG
  ERROR_QUIET
  OUTPUT_STRIP_TRAILING_WHITESPACE  
)

set(FLUID_VERSION_TAG "${FLUID_VERSION_TAG}+sha.${FLUID_VERSION_SHA}.core.sha.${FLUID_CORE_SHA}")

if(result)
  message(VERBOSE "Failed to get version string from Git, falling back to indexed header")
else()
  configure_file("${CMAKE_CURRENT_LIST_DIR}/FluidVersion.hpp.in" "${CMAKE_CURRENT_LIST_DIR}/FluidVersion.hpp" @ONLY)
endif()

set(FLUID_VERSION_PATH ${CMAKE_CURRENT_LIST_DIR})  
