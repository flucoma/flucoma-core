cmake_minimum_required (VERSION 3.8)

# set_property(GLOBAL PROPERTY USE_FOLDERS ON)
project (fluid_decomposition LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
SET(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LIBRARY "libc++")

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic")
IF(APPLE)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "10.7" CACHE STRING "Minimum OS X deployment version")
	set(CMAKE_OSX_ARCHITECTURES x86_64;i386)

  find_library(ACCELERATE Accelerate)
  IF (NOT ACCELERATE)
    message(FATAL_ERROR "Accelerate not found")
  ENDIF()
ENDIF (APPLE)

# Grab the Fluid Decpomposition header files so they can be added to IDE builds
file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/**/*.hpp")

set(root ${CMAKE_CURRENT_SOURCE_DIR})

# HISSTools FFT target: I'd rather this was a static lib; however, this way we
# just inject the source into dependencies downstream, which is unitdy but works
add_library(
  HISSTools_FFT STATIC ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/HISSTools_FFT/HISSTools_FFT.cpp
)
target_sources(HISSTools_FFT INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/HISSTools_FFT/HISSTools_FFT.h
)
target_link_libraries(
  HISSTools_FFT
  PRIVATE ${ACCELERATE}
)

#HISSTools Audiofile Target
add_library(
  HISSTools_AudioFile INTERFACE
)
target_sources(
  HISSTools_AudioFile
  INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/HISSTools_AudioFile/IAudioFile.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/HISSTools_AudioFile/OAudioFile.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/HISSTools_AudioFile/BaseAudioFile.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/HISSTools_AudioFile/IAudioFile.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/HISSTools_AudioFile/OAudioFile.cpp"
)

#Fluid Decomposition headers target
add_library(
  FLUID_DECOMPOSITION INTERFACE
)

target_include_directories(
  FLUID_DECOMPOSITION
  INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(
  FLUID_DECOMPOSITION
  INTERFACE HISSTools_FFT
)

target_sources(
  FLUID_DECOMPOSITION INTERFACE ${HEADERS}
)
target_compile_features(
  FLUID_DECOMPOSITION INTERFACE cxx_std_14
)

#GCC vomits on using HostVector = HostVector<U> without this flag on
if(CMAKE_COMPILER_IS_GNUCXX)
  target_compile_options(
    FLUID_DECOMPOSITION INTERFACE -fpermissive
  )
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

if(APPLE)
target_compile_definitions(
  FLUID_DECOMPOSITION INTERFACE -DEIGEN_USE_BLAS
)
ENDIF(APPLE)
# target_link_libraries(
#   FLUID_DECOMPOSITION INTERFACE HISSTools_FFT
# )

# target_compile_options(
#     FLUID_DECOMPOSITION
#     INTERFACE
#     "$<$<NOT:$<CONFIG:DEBUG>>: -mavx -msse -msse2 -msse3 -msse4>"
# )

target_include_directories(
  FLUID_DECOMPOSITION
  SYSTEM INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty"
)

#add_subdirectory(
#  "${CMAKE_CURRENT_SOURCE_DIR}/examples" EXCLUDE_FROM_ALL
#)
