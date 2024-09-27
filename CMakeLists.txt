# Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
# Copyright University of Huddersfield.
# Licensed under the BSD-3 License.
# See license.md file in the project root for full license information.
# This project has received funding from the European Research Council (ERC)
# under the European Union’s Horizon 2020 research and innovation programme
# (grant agreement No 725899).

cmake_minimum_required (VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
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
  #A consequence of targetting 10.8. Needs to be set globally from 10.15 onwards in order for the test program to compile successfully during configure 
  string(APPEND CMAKE_CXX_FLAGS " -stdlib=libc++")
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

# Either download or point to dependencies
FetchContent_Declare(
  HISSTools
  GIT_REPOSITORY https://github.com/AlexHarker/HISSTools_Library
  GIT_PROGRESS TRUE
  GIT_TAG f3292ad 
)

FetchContent_Declare(
  Eigen
  GIT_SHALLOW TRUE
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_PROGRESS TRUE
  GIT_BRANCH "3.4"
  GIT_TAG "3.4.0"
)

FetchContent_Declare(
  Spectra
  GIT_SHALLOW TRUE
  GIT_REPOSITORY https://github.com/yixuan/spectra
  GIT_PROGRESS TRUE
  GIT_BRANCH "master"
  GIT_TAG "v1.0.1"
)

FetchContent_Declare(
  tl_optional 
  GIT_SHALLOW TRUE
  GIT_REPOSITORY https://github.com/TartanLlama/optional.git  
  GIT_PROGRESS TRUE
)

#see https://json.nlohmann.me/integration/cmake/#fetchcontent
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz)
set(JSON_SystemInclude ON CACHE BOOL "")

FetchContent_Declare(
  Memory
  GIT_SHALLOW TRUE  
  GIT_REPOSITORY https://github.com/foonathan/memory.git
  GIT_PROGRESS TRUE
  GIT_TAG main
)

set(FMT_INSTALL OFF CACHE BOOL "")
FetchContent_Declare(
  fmt
  GIT_SHALLOW TRUE  
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_PROGRESS TRUE
  GIT_TAG master 
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

FetchContent_GetProperties(HISSTools)
if(NOT hisstools_POPULATED)
  FetchContent_Populate(HISSTools)
endif()

FetchContent_GetProperties(Eigen)
if(NOT eigen_POPULATED)
  FetchContent_Populate(Eigen)
endif()

FetchContent_GetProperties(Spectra)
if(NOT spectra_POPULATED)
  FetchContent_Populate(Spectra)
endif()

set(OPTIONAL_BUILD_PACKAGE OFF CACHE BOOL "" FORCE)
set(OPTIONAL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_GetProperties(tl_optional)
if(NOT tl_optional_POPULATED)
  FetchContent_Populate(tl_optional)
endif()

add_subdirectory(${tl_optional_SOURCE_DIR} ${tl_optional_BINARY_DIR} EXCLUDE_FROM_ALL)

FetchContent_GetProperties(json)
if(NOT json_POPULATED)
  FetchContent_Populate(json)  
endif()

add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR} EXCLUDE_FROM_ALL)

set(FOONATHAN_MEMORY_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(FOONATHAN_MEMORY_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(FOONATHAN_MEMORY_BUILD_TESTS OFF CACHE BOOL "" FORCE)

FetchContent_GetProperties(Memory)
if(NOT memory_POPULATED)
  FetchContent_Populate(Memory)
endif()
add_subdirectory(${memory_SOURCE_DIR} ${memory_BINARY_DIR} EXCLUDE_FROM_ALL)

FetchContent_GetProperties(fmt)
if(NOT fmt_POPULATED)
  FetchContent_Populate(fmt)
endif()
add_subdirectory(${fmt_SOURCE_DIR} ${fmt_BINARY_DIR} EXCLUDE_FROM_ALL)

# Brute force staic runtime on windwos
if(MSVC)
  foreach(flag_var
      CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
      CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
    if(${flag_var} MATCHES "/MD")
      string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
    endif()
  endforeach()
endif()


#Fluid Decomposition header-only target
add_library(FLUID_DECOMPOSITION INTERFACE)

target_include_directories(
  FLUID_DECOMPOSITION INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include" 
)
target_include_directories(
  FLUID_DECOMPOSITION SYSTEM INTERFACE #we don't want warnings from Eigen or HissTools
  "${eigen_SOURCE_DIR}"
  "${spectra_SOURCE_DIR}/include"
  "${hisstools_SOURCE_DIR}/include"
  "${memory_SOURCE_DIR}/include/foonathan"
  "${fmt_SOURCE_DIR}/include"
)
target_link_libraries(
  FLUID_DECOMPOSITION INTERFACE 
  flucoma_VERSION_LIB 
  tl::optional
  nlohmann_json::nlohmann_json
  foonathan_memory
  fmt::fmt
)
target_sources(
  FLUID_DECOMPOSITION INTERFACE ${HEADERS}
)

if(MSVC)
  target_compile_definitions(
    FLUID_DECOMPOSITION INTERFACE NOMINMAX _USE_MATH_DEFINES
  )
endif()

#GCC vomits on using HostVector = HostVector<U> without this flag on
if(CMAKE_COMPILER_IS_GNUCXX)
  target_compile_options(FLUID_DECOMPOSITION INTERFACE -fpermissive)
endif()

target_compile_definitions(FLUID_DECOMPOSITION INTERFACE EIGEN_MPL2_ONLY=1)

if(APPLE)
  #targeting <= 10.9, need to really emphasise that we want libc++ both to compiler and linker
  target_compile_options(FLUID_DECOMPOSITION INTERFACE -stdlib=libc++)
  target_link_libraries(FLUID_DECOMPOSITION INTERFACE -stdlib=libc++ ${ACCELERATE})
endif()

#Apply any vector instruction flags
if(DEFINED FLUID_ARCH)
  target_compile_options(FLUID_DECOMPOSITION INTERFACE ${FLUID_ARCH})
endif()

#Examples

option(BUILD_EXAMPLES "Build C++ example code (off by default)" OFF)

if(BUILD_EXAMPLES)
add_subdirectory(
   "${CMAKE_CURRENT_SOURCE_DIR}/examples"
)
endif() 

enable_testing()

if(FLUCOMA_TESTS)
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/tests")
endif()
