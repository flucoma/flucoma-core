# Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
# Copyright 2017-2019 University of Huddersfield.
# Licensed under the BSD-3 License.
# See license.md file in the project root for full license information.
# This project has received funding from the European Research Council (ERC)
# under the European Union’s Horizon 2020 research and innovation programme
# (grant agreement No 725899).

include_guard() 

get_property(isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(isMultiConfig)
  if(NOT "Test" IN_LIST CMAKE_CONFIGURATION_TYPES)
    list(APPEND CMAKE_CONFIGURATION_TYPES Test)
  endif()
else()
  set(allowableBuildTypes Debug Release RelWithDebInfo Test)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY
               STRINGS "${allowableBuildTypes}")
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
  elseif(NOT CMAKE_BUILD_TYPE IN_LIST allowableBuildTypes)
    message(FATAL_ERROR "Invalid build type: ${CMAKE_BUILD_TYPE}")
  endif()
endif()

if(MSVC)
  set(ASSERTION_FLAG "/DNDEBUG")
else()
  set(ASSERTION_FLAG "-DNDEBUG")
endif()

# Take the RelWithDebInfo settings, but remove the NDEBUG flag, so we get assertions
if(CMAKE_C_FLAGS_RELWITHDEBINFO)
  string(REPLACE ${ASSERTION_FLAG} "" CMAKE_C_FLAGS_TEST ${CMAKE_C_FLAGS_RELWITHDEBINFO})
endif()
if(CMAKE_CXX_FLAGS_RELWITHDEBINFO)  
string(REPLACE ${ASSERTION_FLAG} "" CMAKE_CXX_FLAGS_TEST ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
endif()
if(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO)
string(REPLACE ${ASSERTION_FLAG} "" CMAKE_EXE_LINKER_FLAGS_TEST ${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO})
endif()
if(CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO)
  string(REPLACE ${ASSERTION_FLAG} "" CMAKE_SHARED_LINKER_FLAGS_TEST ${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO})
endif()
if(CMAKE_STATIC_LINKER_FLAGS_RELWITHDEBINFO)
  string(REPLACE ${ASSERTION_FLAG} "" CMAKE_STATIC_LINKER_FLAGS_TEST ${CMAKE_STATIC_LINKER_FLAGS_RELWITHDEBINFO})
endif()
if(CMAKE_MODULE_LINKER_FLAGS_RELWITHDEBINFO)
  string(REPLACE ${ASSERTION_FLAG} "" CMAKE_MODULE_LINKER_FLAGS_TEST ${CMAKE_MODULE_LINKER_FLAGS_RELWITHDEBINFO})
endif()
