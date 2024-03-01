foreach (EXAMPLE dataset describe umap)

	add_executable (
			${EXAMPLE} ${EXAMPLE}.cpp
	)

	target_link_libraries(
		${EXAMPLE} PRIVATE FLUID_DECOMPOSITION
	)

	target_compile_options(${EXAMPLE} PRIVATE ${FLUID_ARCH})
	
endforeach (EXAMPLE)
