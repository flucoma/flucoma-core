foreach (EXAMPLE  describe)

	add_executable (
			${EXAMPLE} ${EXAMPLE}.cpp
	)

	target_link_libraries(
		${EXAMPLE} PRIVATE FLUID_DECOMPOSITION  HISSTools_AudioFile HISSTools_FFT
	)

	target_compile_options(${EXAMPLE} PRIVATE ${FLUID_ARCH})
	
endforeach (EXAMPLE)
