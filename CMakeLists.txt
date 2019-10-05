cmake_minimum_required (VERSION 3.8)

project (fluid_decomposition LANGUAGES CXX)
include(FetchContent)
set(HISS_PATH "" CACHE PATH "The path to a HISSTools_Library folder. Will pull from github if not set")

# set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS OFF)
# SET(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LIBRARY "libc++")

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic")
IF(APPLE)
  # set(CMAKE_OSX_DEPLOYMENT_TARGET "10.7" CACHE STRING "Minimum OS X deployment version")
	# set(CMAKE_OSX_ARCHITECTURES x86_64;i386)
  find_library(ACCELERATE Accelerate)
  IF (NOT ACCELERATE)
    message(FATAL_ERROR "Accelerate framework not found")
  ENDIF()
ENDIF (APPLE)

# Grab the Fluid Decpomposition header files so they can be added to IDE builds
file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/**/*.hpp")

if(HISS_PATH)
  get_filename_component(HISS_PATH_FULL ${HISS_PATH} ABSOLUTE)
  message(${HISS_PATH_FULL})
  FetchContent_Declare(
    HISSTools
    DOWNLOAD_COMMAND ""
    SOURCE_DIR "${HISS_PATH_FULL}/"
  )
else()
  FetchContent_Declare(
    HISSTools
    GIT_REPOSITORY https://github.com/AlexHarker/HISSTools_Library
    GIT_PROGRESS TRUE
    GIT_TAG 4d59585e50adc57de6a221b463246193e66d4a5f
  )
endif()

FetchContent_GetProperties(hisstools)
if(NOT hisstools_POPULATED)
  FetchContent_Populate(hisstools)
endif()

# HISSTools FFT target
add_library(
  HISSTools_FFT STATIC "${hisstools_SOURCE_DIR}/HISSTools_FFT/HISSTools_FFT.cpp"
)
target_sources(HISSTools_FFT INTERFACE
  "${hisstools_SOURCE_DIR}/HISSTools_FFT/HISSTools_FFT.h"
)
target_link_libraries(
  HISSTools_FFT PRIVATE ${ACCELERATE}
)

target_include_directories(HISSTools_FFT PUBLIC "${hisstools_SOURCE_DIR}")

set_target_properties(HISSTools_FFT
  PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

add_library(HISSTools_SIMD INTERFACE)
target_sources(HISSTools_SIMD INTERFACE "${hisstools_SOURCE_DIR}/SIMDSupport.hpp")

#HISSTools Audiofile Target
add_library(
  HISSTools_AudioFile
  STATIC
  "${hisstools_SOURCE_DIR}/AudioFile/BaseAudioFile.cpp"
  "${hisstools_SOURCE_DIR}/AudioFile/IAudioFile.cpp"
  "${hisstools_SOURCE_DIR}/AudioFile/OAudioFile.cpp"
)
target_sources(
  HISSTools_AudioFile
  INTERFACE
  "${hisstools_SOURCE_DIR}/AudioFile/IAudioFile.h"
  "${hisstools_SOURCE_DIR}/AudioFile/OAudioFile.h"
)

set_target_properties(HISSTools_AudioFile
  PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)


#Fluid Decomposition header-only target
add_library(FLUID_DECOMPOSITION INTERFACE)

target_include_directories(
  FLUID_DECOMPOSITION INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_include_directories(
  FLUID_DECOMPOSITION SYSTEM INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty"
)

target_link_libraries(
  FLUID_DECOMPOSITION INTERFACE HISSTools_FFT HISSTools_SIMD
)

target_sources(
  FLUID_DECOMPOSITION INTERFACE ${HEADERS}
)

target_compile_features(
  FLUID_DECOMPOSITION INTERFACE cxx_std_14 $<$<PLATFORM_ID:APPLE>:-stdlib=libc++>
)

#GCC vomits on using HostVector = HostVector<U> without this flag on
if(CMAKE_COMPILER_IS_GNUCXX)
  target_compile_options(FLUID_DECOMPOSITION INTERFACE -fpermissive)
endif()

if(APPLE)
  target_compile_definitions(FLUID_DECOMPOSITION INTERFACE -DEIGEN_USE_BLAS)
endif()
