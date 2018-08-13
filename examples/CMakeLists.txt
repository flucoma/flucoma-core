cmake_minimum_required (VERSION 3.0.0)
project (examples LANGUAGES CXX)

find_path(SNDFILE_INCLUDE_DIR sndfile.h
	HINTS /usr/local/opt/libsndfile/include
)
find_library(SNDFILE_LIBRARY NAMES libsndfile.dylib
	HINTS /usr/local/opt/libsndfile/lib
)

# common to all examples
foreach ( EXAMPLE  audiofile;fluidtensor;stft;nmf;sineextraction)
	add_executable(
		test_${EXAMPLE}
		test_${EXAMPLE}.cpp
	)
	target_include_directories(
		test_${EXAMPLE} PRIVATE
		${PROJECT_SOURCE_DIR}/../include
		${PROJECT_SOURCE_DIR}/../3rdparty
		${SNDFILE_INCLUDE_DIR}
	)
	set_target_properties(test_${EXAMPLE} PROPERTIES CXX_STANDARD 11)
	target_compile_options(test_${EXAMPLE} PRIVATE -O3 -mavx -DEIGEN_USE_BLAS)
	target_link_libraries (test_${EXAMPLE} ${SNDFILE_LIBRARY})
endforeach(EXAMPLE)

# required for examples that use fft
foreach ( FFT_EXAMPLE  stft;nmf;sineextraction)
	target_sources(
		test_${FFT_EXAMPLE} PRIVATE
		${PROJECT_SOURCE_DIR}/../3rdparty/HISSTools_FFT/HISSTools_FFT.cpp
	)
	target_link_libraries (test_${FFT_EXAMPLE} "-framework Accelerate")
endforeach(FFT_EXAMPLE)
