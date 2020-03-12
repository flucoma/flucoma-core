# Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
# Copyright 2017-2019 University of Huddersfield.
# Licensed under the BSD-3 License.
# See license.md file in the project root for full license information.
# This project has received funding from the European Research Council (ERC)
# under the European Union’s Horizon 2020 research and innovation programme
# (grant agreement No 725899).

include_guard()

find_package(Git REQUIRED)

execute_process( 
  COMMAND ${GIT_EXECUTABLE} describe --abbrev=0 --always
  RESULT_VARIABLE result
  OUTPUT_VARIABLE FLUID_VERSION_TAG
  OUTPUT_STRIP_TRAILING_WHITESPACE  
)

if(result)
  message(FATAL_ERROR "Failed to get version string from Git")
endif()

configure_file("${CMAKE_CURRENT_LIST_DIR}/FluidVersion.hpp.in" "${CMAKE_CURRENT_LIST_DIR}/FluidVersion.hpp" @ONLY)
set(FLUID_VERSION_PATH ${CMAKE_CURRENT_LIST_DIR})  
