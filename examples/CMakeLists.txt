cmake_minimum_required (VERSION 3.0.0)
project (examples LANGUAGES CXX)

find_path(SNDFILE_INCLUDE_DIR sndfile.h
	HINTS /usr/local/opt/libsndfile/include
)
find_library(SNDFILE_LIBRARY NAMES libsndfile.dylib
	HINTS /usr/local/opt/libsndfile/lib
)


file(GLOB LIBRARY_HEADERS ${PROJECT_SOURCE_DIR}/../include/*/*.hpp)

# common to all examples
foreach ( EXAMPLE  audiofile;fluidtensor;stft;nmf;fluidbuffer;sineextraction;transient_extractor;eigen_return_speed)
	add_executable(
		test_${EXAMPLE}
		test_${EXAMPLE}.cpp
		${LIBRARY_HEADERS} #add headers for editing in Xcode
	)
	target_include_directories(
		test_${EXAMPLE} PRIVATE
		${PROJECT_SOURCE_DIR}/../include
		${PROJECT_SOURCE_DIR}/../3rdparty
		${SNDFILE_INCLUDE_DIR}
	)
	set_target_properties(test_${EXAMPLE} PROPERTIES CXX_STANDARD 11)
	target_compile_options(test_${EXAMPLE} PRIVATE -O3 -mavx)
	target_compile_options(test_${EXAMPLE} PRIVATE "$<$<CONFIG:RELEASE>:-DEIGEN_USE_BLAS>")
	target_link_libraries (test_${EXAMPLE} ${SNDFILE_LIBRARY})
endforeach(EXAMPLE)

# required for examples that use fft
foreach ( FFT_EXAMPLE  stft;nmf;sineextraction;transient_extractor)
	target_sources(
		test_${FFT_EXAMPLE} PRIVATE
		${PROJECT_SOURCE_DIR}/../3rdparty/HISSTools_FFT/HISSTools_FFT.cpp
	)
	target_link_libraries (test_${FFT_EXAMPLE} "-framework Accelerate")
endforeach(FFT_EXAMPLE)

target_sources(
	test_transient_extractor PRIVATE
	${PROJECT_SOURCE_DIR}/../3rdparty/HISSTools_AudioFile/BaseAudioFile.cpp
	${PROJECT_SOURCE_DIR}/../3rdparty/HISSTools_AudioFile/IAudioFile.cpp
	${PROJECT_SOURCE_DIR}/../3rdparty/HISSTools_AudioFile/OAudioFile.cpp
)
