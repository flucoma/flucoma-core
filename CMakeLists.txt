# Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
# Copyright University of Huddersfield.
# Licensed under the BSD-3 License.
# See license.md file in the project root for full license information.
# This project has received funding from the European Research Council (ERC)
# under the European Union’s Horizon 2020 research and innovation programme
# (grant agreement No 725899).

cmake_minimum_required (VERSION 3.28)

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.28.0")
  cmake_policy(SET CMP0135 NEW)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/script")

#set module path at top level so wrapper projects can easily include fluid_version script
get_directory_property(hasParent PARENT_DIRECTORY)
if(hasParent)  
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/script"
  PARENT_SCOPE)
endif()

if(APPLE)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "10.9" CACHE STRING "")
endif()

project (flucoma-core LANGUAGES CXX)

include(flucoma-buildtype)
include(FlucomaVersion.cmake)
include(FlucomaClients.cmake)

include(FetchContent)
set(HISS_PATH "" CACHE PATH "The path to a HISSTools_Library folder. Will pull from github if not set")
set(EIGEN_PATH "" CACHE PATH "The path to an Eigen installation (>=3.3.5). Will pull from github if not set")
set(SPECTRA_PATH "" CACHE PATH "The path to aa Spectra installation. Will pull from github if not set")
IF(APPLE)
  find_library(ACCELERATE Accelerate)
  IF (NOT ACCELERATE)
    message(FATAL_ERROR "Accelerate framework not found")
  ENDIF()
ENDIF (APPLE)

# Grab the Fluid Decpomposition header files so they can be added to IDE builds
file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/**/*.hpp")

############################################################################################################
# Dependencies
############################################################################################################
FetchContent_Declare(
  HISSTools
  GIT_REPOSITORY https://github.com/AlexHarker/HISSTools_Library
  GIT_PROGRESS TRUE
  GIT_TAG f3292ad 
  EXCLUDE_FROM_ALL
)

FetchContent_Declare(
  Eigen
  GIT_SHALLOW TRUE
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_PROGRESS TRUE
  GIT_BRANCH "3.4"
  GIT_TAG "3.4.0"
  SOURCE_SUBDIR "include" # eigen's cmake doesn't do anything useful for us, skip it
  EXCLUDE_FROM_ALL
)

FetchContent_Declare(
  Spectra
  GIT_SHALLOW TRUE
  GIT_REPOSITORY https://github.com/yixuan/spectra
  GIT_PROGRESS TRUE
  GIT_BRANCH "master"
  GIT_TAG "v1.0.1"
  SOURCE_SUBDIR "include" #spectra behaves badly as cmake sub-project, so skip its cmake
  EXCLUDE_FROM_ALL
)

FetchContent_Declare(
  tl_optional 
  GIT_SHALLOW TRUE
  GIT_REPOSITORY https://github.com/TartanLlama/optional.git  
  GIT_PROGRESS TRUE
  EXCLUDE_FROM_ALL
)
set(OPTIONAL_BUILD_PACKAGE OFF CACHE BOOL "" FORCE)
set(OPTIONAL_BUILD_TESTS OFF CACHE BOOL "" FORCE)

#see https://json.nlohmann.me/integration/cmake/#fetchcontent
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
set(JSON_SystemInclude ON CACHE BOOL "")

FetchContent_Declare(
  Memory
  GIT_SHALLOW TRUE  
  GIT_REPOSITORY https://github.com/foonathan/memory.git
  GIT_PROGRESS TRUE
  GIT_TAG main
  EXCLUDE_FROM_ALL
)

set(FMT_INSTALL OFF CACHE BOOL "")
FetchContent_Declare(
  fmt
  GIT_SHALLOW TRUE  
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_PROGRESS TRUE
  GIT_TAG master 
  EXCLUDE_FROM_ALL
)

if(HISS_PATH) #if hiss path is set, this will stop it downloading
  get_filename_component(FETCHCONTENT_SOURCE_DIR_HISSTOOLS ${HISS_PATH} ABSOLUTE)
endif()

if(EIGEN_PATH) #if eigen path is set, this will stop it downloading
  get_filename_component(FETCHCONTENT_SOURCE_DIR_EIGEN ${EIGEN_PATH} ABSOLUTE)
endif()

if(SPECTRA_PATH) #if spectra path is set, this will stop it downloading
  get_filename_component(FETCHCONTENT_SOURCE_DIR_SPECTRA ${SPECTRA_PATH} ABSOLUTE)
endif()

if(JSON_PATH)
  get_filename_component(FETCHCONTENT_SOURCE_DIR_JSON ${JSON_PATH} ABSOLUTE)  
endif()

FetchContent_MakeAvailable(HISSTools Eigen Spectra tl_optional json Memory fmt)

set(FOONATHAN_MEMORY_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(FOONATHAN_MEMORY_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(FOONATHAN_MEMORY_BUILD_TESTS OFF CACHE BOOL "" FORCE)


############################################################################################################
# flucoma-core target, gathers dependencies, our headers and transitive build requirements 
############################################################################################################
add_library(FLUID_DECOMPOSITION INTERFACE)

# this should pass on to all consumers of this target 
target_compile_features(FLUID_DECOMPOSITION INTERFACE cxx_std_17)

target_include_directories(FLUID_DECOMPOSITION INTERFACE 
  "${CMAKE_CURRENT_SOURCE_DIR}/include" 
)

target_include_directories(FLUID_DECOMPOSITION SYSTEM INTERFACE 
  "${eigen_SOURCE_DIR}"
  "${spectra_SOURCE_DIR}/include"
  "${hisstools_SOURCE_DIR}/include"
  "${memory_SOURCE_DIR}/include/foonathan"
  "${fmt_SOURCE_DIR}/include"
)

target_link_libraries(FLUID_DECOMPOSITION INTERFACE 
  flucoma_VERSION_LIB 
  tl::optional
  nlohmann_json::nlohmann_json
  foonathan_memory
  fmt::fmt
  $<$<PLATFORM_ID:Darwin>:${ACCELERATE}>
)
target_sources(
  FLUID_DECOMPOSITION INTERFACE ${HEADERS}
)

target_compile_definitions(FLUID_DECOMPOSITION INTERFACE
  EIGEN_MPL2_ONLY=1
  $<$<CXX_COMPILER_ID:MSVC>:FLUID_DECOMPOSITION INTERFACE NOMINMAX _USE_MATH_DEFINES>
)

#GCC vomits on using HostVector = HostVector<U> without this flag on
target_compile_options(FLUID_DECOMPOSITION INTERFACE 
  $<$<CXX_COMPILER_ID:Gnu>:-fpermissive>
)

############################################################################################################
# Examples
############################################################################################################
option(BUILD_EXAMPLES "Build C++ example code (off by default)" OFF)

if(BUILD_EXAMPLES)
add_subdirectory(
   "${CMAKE_CURRENT_SOURCE_DIR}/examples"
)
endif() 

############################################################################################################
# Tests 
############################################################################################################
enable_testing()

if(FLUCOMA_TESTS)
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/tests")
endif()
