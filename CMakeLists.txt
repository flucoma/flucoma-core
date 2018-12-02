cmake_minimum_required (VERSION 3.3)
project (fluid_decomposition LANGUAGES CXX)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
SET(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LIBRARY "libc++")
set(CMAKE_OSX_DEPLOYMENT_TARGET "10.7" CACHE STRING "Minimum OS X deployment version")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic")
IF(APPLE)
  find_library(ACCELERATE Accelerate)
  IF (NOT ACCELERATE)
    message(FATAL_ERROR "Accelerate not found")
  ENDIF()
ENDIF (APPLE)

### HISSTools FFT target
add_library(
  HISSTools_FFT INTERFACE
)
target_sources(
  HISSTools_FFT
  INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/HISSTools_FFT/HISSTools_FFT.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/HISSTools_FFT/HISSTools_FFT.cpp"
)
target_link_libraries(
  HISSTools_FFT
  INTERFACE ${ACCELERATE}
)

# set_target_properties(
#   HISSTools_FFT PROPERTIES FOLDER ThirdParty/FFT
# )

#HISSTools Audiofile Target
add_library(
  HISSTools_AudioFile
  "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/HISSTools_AudioFile/IAudioFile.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/HISSTools_AudioFile/OAudioFile.h"
)
target_sources(
  HISSTools_AudioFile
  PRIVATE
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

target_include_directories(
  FLUID_DECOMPOSITION
  SYSTEM INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty"
)


#TODO: Not this? We possibly need to seperate public facing headers from
# obviously implementaion related ones to cut down on noise
file(GLOB_RECURSE LIBRARY_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*/*.hpp)

target_sources(
  FLUID_DECOMPOSITION
  INTERFACE ${LIBRARY_HEADERS}
)
target_compile_features(
  FLUID_DECOMPOSITION INTERFACE cxx_std_14
)
target_compile_definitions(
  FLUID_DECOMPOSITION INTERFACE -DEIGEN_USE_BLAS
)
target_link_libraries(
  FLUID_DECOMPOSITION INTERFACE HISSTools_FFT
)
add_subdirectory(
  "${CMAKE_CURRENT_SOURCE_DIR}/examples"
)

export(TARGETS FLUID_DECOMPOSITION HISSTools_FFT FILE fluid_decomposition-exports.cmake)
