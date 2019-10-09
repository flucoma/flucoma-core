cmake_minimum_required (VERSION 3.11)

project (fluid_decomposition LANGUAGES CXX)
include(FetchContent)
set(HISS_PATH "" CACHE PATH "The path to a HISSTools_Library folder. Will pull from github if not set")
set(EIGEN_PATH "" CACHE PATH "The path to an Eigen installation (>=3.3.5). Will pull from github if not set")
IF(APPLE)
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

if(EIGEN_PATH)
  get_filename_component(EIGEN_PATH_FULL ${EIGEN_PATH} ABSOLUTE)
  message(STATUS ${EIGEN_PATH_FULL})
  FetchContent_Declare(
    Eigen
    DOWNLOAD_COMMAND ""
    SOURCE_DIR "${EIGEN_PATH_FULL}/"
  )
else()
  FetchContent_Declare(
    Eigen
    GIT_REPOSITORY https://github.com/eigenteam/eigen-git-mirror/
    GIT_PROGRESS TRUE
    GIT_BRANCH "branches/3.3"
    GIT_TAG "3.3.5"
  )
endif()

FetchContent_GetProperties(hisstools)
if(NOT hisstools_POPULATED)
  FetchContent_Populate(hisstools)
endif()

FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
  FetchContent_Populate(eigen)
endif()

# HISSTools FFT target
add_library(
  HISSTools_FFT STATIC "${hisstools_SOURCE_DIR}/HISSTools_FFT/HISSTools_FFT.cpp"
)

target_link_libraries(
  HISSTools_FFT PRIVATE ${ACCELERATE}
)

target_include_directories(HISSTools_FFT PUBLIC "${hisstools_SOURCE_DIR}")

set_target_properties(HISSTools_FFT PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

#HISSTools Audiofile Target
add_library(
  HISSTools_AudioFile STATIC
  "${hisstools_SOURCE_DIR}/AudioFile/BaseAudioFile.cpp"
  "${hisstools_SOURCE_DIR}/AudioFile/IAudioFile.cpp"
  "${hisstools_SOURCE_DIR}/AudioFile/OAudioFile.cpp"
)

set_target_properties(HISSTools_AudioFile PROPERTIES
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
  FLUID_DECOMPOSITION SYSTEM INTERFACE #we don't want warnings from Eigen or HissTools
  "${eigen_SOURCE_DIR}"
  "${hisstools_SOURCE_DIR}"
)
target_link_libraries(
  FLUID_DECOMPOSITION INTERFACE HISSTools_FFT
)
target_sources(
  FLUID_DECOMPOSITION INTERFACE ${HEADERS}
)

#GCC vomits on using HostVector = HostVector<U> without this flag on
if(CMAKE_COMPILER_IS_GNUCXX)
  target_compile_options(FLUID_DECOMPOSITION INTERFACE -fpermissive)
endif()

if(APPLE)
  target_compile_definitions(FLUID_DECOMPOSITION INTERFACE -DEIGEN_USE_BLAS)
endif()
