cmake_minimum_required (VERSION 2.6)
project (examples)
set(CMAKE_BUILD_TYPE "Release")

find_path(SNDFILE_INCLUDE_DIR sndfile.h
	HINTS /usr/local/opt/libsndfile/include
)

find_library(SNDFILE_LIBRARY NAMES libsndfile.dylib
	HINTS /usr/local/opt/libsndfile/lib
)

set(THIRDPARTY ${PROJECT_SOURCE_DIR}/../3rdparty)
set(LIBRARY_ROOT ${PROJECT_SOURCE_DIR}/../include)

include_directories(
	${LIBRARY_ROOT}
	${THIRDPARTY}
	${SNDFILE_INCLUDE_DIR}
)

set(CMAKE_CXX_FLAGS "-Wall -std=c++11 -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -mavx -DEIGEN_USE_BLAS")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g -mavx")


add_executable(test_audiofile
	test_audiofile.cpp
)
target_link_libraries (test_audiofile ${SNDFILE_LIBRARY})


add_executable(test_tensor
	TensorTests.cpp
)
target_link_libraries (test_tensor ${SNDFILE_LIBRARY})


add_executable(test_stft
	${THIRDPARTY}/HISSTools_FFT/HISSTools_FFT.cpp
	test_stft.cpp
)
target_link_libraries (test_stft ${SNDFILE_LIBRARY})
target_link_libraries (test_stft "-framework Accelerate")

add_executable(test_nmf
	${THIRDPARTY}/HISSTools_FFT/HISSTools_FFT.cpp
	test_nmf.cpp
)
target_link_libraries (test_nmf ${SNDFILE_LIBRARY})
target_link_libraries (test_nmf "-framework Accelerate")

add_executable(test_sineextraction
	${THIRDPARTY}/HISSTools_FFT/HISSTools_FFT.cpp
	test_sineextraction.cpp
)
target_link_libraries (test_sineextraction ${SNDFILE_LIBRARY})
target_link_libraries (test_sineextraction "-framework Accelerate")
