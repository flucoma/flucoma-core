cmake_minimum_required (VERSION 3.11)

if(APPLE) #has to be before first project() call. grrrrrr
  set(CMAKE_OSX_DEPLOYMENT_TARGET 10.7)  
endif()

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

set(FETCHCONTENT_QUIET FALSE)

# Either download or point to dependencies
FetchContent_Declare(
  HISSTools
  GIT_REPOSITORY https://github.com/AlexHarker/HISSTools_Library
  GIT_PROGRESS TRUE
  GIT_TAG 9d9a85f
)

FetchContent_Declare(
  Eigen
  GIT_REPOSITORY https://github.com/eigenteam/eigen-git-mirror/
  GIT_PROGRESS TRUE
  GIT_BRANCH "branches/3.3"
  GIT_TAG "3.3.5"
)

if(HISS_PATH) #if hiss path is set, this will stop it downloading
  get_filename_component(FETCHCONTENT_SOURCE_DIR_HISSTOOLS ${HISS_PATH} ABSOLUTE)
endif()

if(EIGEN_PATH) #if hiss path is set, this will stop it downloading
  get_filename_component(FETCHCONTENT_SOURCE_DIR_EIGEN ${EIGEN_PATH} ABSOLUTE)
endif()

FetchContent_GetProperties(HISSTools)
if(NOT hisstools_POPULATED)
  FetchContent_Populate(HISSTools)
endif()

FetchContent_GetProperties(Eigen)
if(NOT eigen_POPULATED)
  FetchContent_Populate(Eigen)
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
  #targeting <= 10.9, need to really emphasise that we want libc++ both to compiler and linker
  target_compile_options(HISSTools_FFT PUBLIC -stdlib=libc++)
  target_compile_options(HISSTools_AudioFile PUBLIC -stdlib=libc++)
  target_compile_options(FLUID_DECOMPOSITION INTERFACE -stdlib=libc++)
  target_link_libraries(FLUID_DECOMPOSITION INTERFACE -stdlib=libc++)
endif()

#Examples 
if(APPLE)
  add_subdirectory(
   "${CMAKE_CURRENT_SOURCE_DIR}/examples" EXCLUDE_FROM_ALL
  )
endif() 
