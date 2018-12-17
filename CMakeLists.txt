cmake_minimum_required (VERSION 3.8)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
project (fluid_decomposition LANGUAGES CXX)
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

set(HEADERS "")
list(APPEND HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/public/HPSS.hpp"
include/algorithms/public/NMF.hpp
include/algorithms/public/NoveltySegmentation.hpp
include/algorithms/public/OnsetSegmentation.hpp
include/algorithms/public/RTHPSS.hpp
include/algorithms/public/RTSineExtraction.hpp
include/algorithms/public/RatioMask.hpp
include/algorithms/public/STFT.hpp
include/algorithms/public/SineExtraction.hpp
include/algorithms/public/TransientExtraction.hpp
include/algorithms/public/TransientSegmentation.hpp
include/algorithms/public/Windows.hpp
include/clients/common/AudioClient.hpp
include/clients/common/OfflineClient.hpp
include/clients/common/FluidBaseClient.hpp
include/clients/common/FluidParams.hpp
include/clients/common/FluidSink.hpp
include/clients/common/FluidSource.hpp
include/clients/common/ParameterConstraints.hpp
include/clients/common/ParameterDescriptor.hpp
include/clients/common/ParameterDescriptorList.hpp
include/clients/common/ParameterInstanceList.hpp
include/clients/common/ParameterTypes.hpp
include/clients/common/STFTCheckParams.hpp
include/clients/nrt/BufferComposeNRT.hpp
include/clients/nrt/FluidNRTClientWrapper.hpp
include/clients/nrt/HPSS.hpp
include/clients/nrt/NMFClient.hpp
include/clients/nrt/NoveltyClient.hpp
include/clients/nrt/Sines.hpp
include/clients/nrt/TransientNRTClient.hpp
include/clients/nrt/TransientSliceNRT.hpp
include/clients/rt/BaseAudioClient.hpp
include/clients/rt/BaseSTFTClient.hpp
include/clients/rt/BufferedProcess.hpp
include/clients/rt/GainClient.hpp
include/clients/rt/HPSSClient.hpp
include/clients/rt/NMFMatch.hpp
include/clients/rt/SinesClient.hpp
include/clients/rt/TransientClient.hpp
include/clients/rt/TransientSlice.hpp )

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/include  FILES ${HEADERS})
source_group(3rdparty FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/HISSTools_FFT/HISSTools_FFT.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/HISSTools_FFT/HISSTools_FFT.cpp"
)


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
file(GLOB_RECURSE LIBRARY_HEADERS include/*/*.hpp)



target_sources(
  FLUID_DECOMPOSITION
  INTERFACE
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/public/HPSS.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/public/NMF.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/public/NoveltySegmentation.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/public/OnsetSegmentation.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/public/RTHPSS.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/public/RTSineExtraction.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/public/RatioMask.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/public/STFT.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/public/SineExtraction.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/public/TransientExtraction.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/public/TransientSegmentation.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/public/Windows.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/common/AudioClient.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/common/OfflineClient.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/common/BufferAdaptor.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/common/FluidBaseClient.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/common/FluidParams.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/common/FluidSink.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/common/FluidSource.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/common/ParameterConstraints.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/common/ParameterDescriptor.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/common/ParameterDescriptorList.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/common/ParameterInstanceList.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/common/ParameterTypes.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/common/STFTCheckParams.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/nrt/BufferComposeNRT.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/nrt/FluidNRTClientWrapper.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/nrt/HPSS.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/nrt/NMFClient.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/nrt/NoveltyClient.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/nrt/Sines.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/nrt/TransientNRTClient.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/nrt/TransientSliceNRT.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/rt/BaseAudioClient.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/rt/BaseSTFTClient.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/rt/BufferedProcess.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/rt/GainClient.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/rt/HPSSClient.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/rt/NMFMatch.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/rt/SinesClient.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/rt/TransientClient.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/rt/TransientSlice.hpp
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
