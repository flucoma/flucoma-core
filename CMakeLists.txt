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

# TODO: Break into sub CMakeLists for algos / clients etc to avoid the list of doooom
list(APPEND HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/public/CepstrumF0.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/public/EnvelopeSegmentation.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/public/HPS.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/public/HPSS.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/public/Loudness.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/public/NMF.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/public/NoveltySegmentation.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/public/OnsetDetectionFuncs.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/public/OnsetSegmentation.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/public/RTHPSS.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/public/RTNoveltySegmentation.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/public/RTSineExtraction.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/public/RatioMask.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/public/STFT.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/public/SineExtraction.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/public/SpectralShape.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/public/Stats.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/public/TransientExtraction.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/public/TransientSegmentation.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/public/WindowFuncs.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/public/Windows.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/public/YINFFT.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/util/ARModel.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/util/ButterworthHPFilter.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/util/ConvolutionTools.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/util/DCT.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/util/Descriptors.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/util/FFT.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/util/FluidEigenMappings.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/util/KWeightingFilter.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/util/MedianFilter.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/util/MelBands.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/util/Novelty.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/util/PeakDetection.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/util/RTNovelty.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/util/SlideUDFilter.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/util/Toeplitz.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/algorithms/util/TruePeak.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/common/AudioClient.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/common/BufferAdaptor.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/common/FluidBaseClient.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/common/FluidSink.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/common/FluidSource.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/common/OfflineClient.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/common/ParameterConstraints.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/common/ParameterSet.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/common/ParameterTrackChanges.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/common/ParameterTypes.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/common/Result.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/common/SpikesToTimes.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/common/TupleUtilities.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/nrt/BufferComposeNRT.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/nrt/BufferStats.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/nrt/FluidNRTClientWrapper.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/nrt/NMFClient.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/rt/AmpSlice.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/rt/BaseSTFTClient.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/rt/BufferedProcess.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/rt/GainClient.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/rt/HPSSClient.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/rt/LoudnessClient.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/rt/MFCCClient.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/rt/MelBandsClient.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/rt/NMFFilter.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/rt/NMFMatch.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/rt/NoveltySlice.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/rt/OnsetSlice.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/rt/PitchClient.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/rt/SinesClient.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/rt/SpectralShapeClient.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/rt/TransientClient.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/clients/rt/TransientSlice.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/data/FluidMeta.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/data/FluidTensor.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/data/FluidTensor_Support.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/data/TensorTypes.hpp
${CMAKE_CURRENT_SOURCE_DIR}/include/util/TupleUtils.hpp )

set(root ${CMAKE_CURRENT_SOURCE_DIR})

# HISSTools FFT target: I'd rather this was a static lib; however, this way we
# just inject the source into dependencies downstream, which is unitdy but works
add_library(
  HISSTools_FFT INTERFACE
)
target_sources(HISSTools_FFT INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/HISSTools_FFT/HISSTools_FFT.h
  ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/HISSTools_FFT/HISSTools_FFT.cpp
)
target_link_libraries(
  HISSTools_FFT
  INTERFACE ${ACCELERATE}
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

export(TARGETS FLUID_DECOMPOSITION HISSTools_FFT HISSTools_AudioFile FILE fluid_decomposition-exports.cmake)
