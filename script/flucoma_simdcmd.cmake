# Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
# Copyright 2017-2019 University of Huddersfield.
# Licensed under the BSD-3 License.
# See license.md file in the project root for full license information.
# This project has received funding from the European Research Council (ERC)
# under the European Union’s Horizon 2020 research and innovation programme
# (grant agreement No 725899).

include_guard()

if(APPLE)
    # https://stackoverflow.com/a/45921250
    set (SIMD_OPT -march=core2 -mtune=haswell)  
elseif(MSVC)
    if(CMAKE_SIZEOF_VOID_P EQUAL 4) #32bit; SSE2 is always on for x64 MSVC
      set(SIMD_OPT /arch:SSE2)
    endif()
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "amd64.*|x86_64.*|AMD64.*|i686.*|i386.*|x86.*")
        set(SIMD_OPT -msse4)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")   
       set(SIMD_OPT -march=armv7-a -mtune=cortex-a8 -mfloat-abi=hard -mfpu=neon)
    endif() 
else()
      message(WARNING "Don't know about ${CMAKE_SYSTEM_PROCESSOR} type: if you know the compiler flag for enabling vector instructions, please pass this to CMake with -DFLUID_ARCH")
endif() 

if(SIMD_OPT)
  set(FLUID_ARCH ${SIMD_OPT} CACHE STRING "Flag for using vector instruction sets")
endif()
