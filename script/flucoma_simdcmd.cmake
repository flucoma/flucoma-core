# Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
# Copyright 2017-2019 University of Huddersfield.
# Licensed under the BSD-3 License.
# See license.md file in the project root for full license information.
# This project has received funding from the European Research Council (ERC)
# under the European Union’s Horizon 2020 research and innovation programme
# (grant agreement No 725899).

include_guard()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "amd64.*|x86_64.*|AMD64.*|i686.*|i386.*|x86.*")
  if(MSVC)
    set(SIMD_OPT /ARCH:AVX)
  else()
    set(SIMD_OPT -mavx)
  endif() 
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")   
   set(SIMD_OPT mfloat-abi=hard -mfpu=neon) 
else() 
    message(WARNING "Don't know about ${CMAKE_SYSTEM_PROCESSOR} type: if you know the compiler flag for enabling vector instructions, please pass this to CMake with -DFLUID_ARCH")
endif()

if(SIMD_OPT)
set(FLUID_ARCH ${SIMD_OPT} CACHE STRING "Flag for using vector instruction sets")
endif()
