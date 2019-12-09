cmake_minimum_required (VERSION 3.11)
# project (examples LANGUAGES CXX)

find_path(SNDFILE_INCLUDE_DIR sndfile.h
	HINTS /usr/local/opt/libsndfile/include
)
find_library(SNDFILE_LIBRARY NAMES libsndfile.dylib
	HINTS /usr/local/opt/libsndfile/lib
)

foreach (EXAMPLE  audiofile;fluidtensor;stft;nmf;
									fluidbuffer;sineextraction;transient_extractor;
								median_filter;rthpss;rtsineextraction;novelty;paramDescriptorList;onset;transient_segmentation;envseg)

	add_executable (
			test_${EXAMPLE} test_${EXAMPLE}.cpp
	)
	target_include_directories(
	  test_${EXAMPLE}
		PRIVATE ${SNDFILE_INCLUDE_DIR}
	)
	target_link_libraries(
		test_${EXAMPLE} PRIVATE FLUID_DECOMPOSITION  ${SNDFILE_LIBRARY} HISSTools_FFT
	)
	target_compile_options(
		test_${EXAMPLE} PRIVATE "$<$<NOT:$<CONFIG:DEBUG>>:-mavx>"
	)
  source_group(TREE  ${root}  FILES ${HEADERS})


endforeach (EXAMPLE)

# file(GLOB LIBRARY_HEADERS ${PROJECT_SOURCE_DIR}/../include/*/*.hpp)
#
# # common to all examples
# foreach ( EXAMPLE  audiofile;fluidtensor;stft;nmf;fluidbuffer;sineextraction;transient_extractor)
# 	add_executable(
# 		test_${EXAMPLE}
# 		test_${EXAMPLE}.cpp
# 		${LIBRARY_HEADERS} #add headers for editing in Xcode
# 	)
# 	target_include_directories(
# 		test_${EXAMPLE} PRIVATE
# 		${PROJECT_SOURCE_DIR}/../include
# 		${PROJECT_SOURCE_DIR}/../3rdparty
# 		${SNDFILE_INCLUDE_DIR}
# 	)
# 	set_target_properties(test_${EXAMPLE} PROPERTIES CXX_STANDARD 11)
# 	target_compile_options(test_${EXAMPLE} PRIVATE -O3 -mavx)
# 	target_compile_options(test_${EXAMPLE} PRIVATE "$<$<CONFIG:RELEASE>:-DEIGEN_USE_BLAS>")
# 	target_link_libraries (test_${EXAMPLE} ${SNDFILE_LIBRARY})
# endforeach(EXAMPLE)
#
# required for examples that use fft
# foreach ( FFT_EXAMPLE  stft;nmf;sineextraction;transient_extractor)
# 	target_link_libraries (test_${FFT_EXAMPLE} PRIVATE HISSTools_FFT)
# endforeach(FFT_EXAMPLE)
#
target_link_libraries(
	test_transient_extractor PRIVATE HISSTools_AudioFile
)

target_link_libraries(
	test_transient_segmentation PRIVATE HISSTools_AudioFile
)

target_link_libraries(
	test_envseg PRIVATE HISSTools_AudioFile
)
