
file(REMOVE_RECURSE "${CMAKE_BINARY_DIR}/generated_sources") 

function(pr_generate_includes headers var) 
  foreach(h ${headers})
    string(APPEND include_block "#include <${h}>\n")
  endforeach()
  set(${var} ${include_block} PARENT_SCOPE)
endfunction() 

function(pr_generate_makeclass clients headers classes template var externals_out)
  foreach(client header class IN ZIP_LISTS clients headers classes)
    make_external_name(${client} ${header} external)
    list(APPEND externals ${external})
    string(CONFIGURE "${template}" make_class)
    string(APPEND make_class_block "  ${make_class}\n")    
  endforeach() 
  set(${var} ${make_class_block} PARENT_SCOPE)  
  set(${externals_out} ${externals} PARENT_SCOPE)
endfunction()


function(generate_source)  
  # Define the supported set of keywords
  set(noValues "")
  set(singleValues FILENAME EXTRA_SOURCE EXTERNALS_OUT FILE_OUT)
  set(multiValues CLIENTS HEADERS CLASSES)
  # Process the arguments passed in
  include(CMakeParseArguments)
  cmake_parse_arguments(ARG
  "${noValues}"
  "${singleValues}"
  "${multiValues}"
  ${ARGN})  
    
  pr_generate_includes("${ARG_HEADERS}" INCLUDE_BLOCK)
  pr_generate_makeclass("${ARG_CLIENTS}" "${ARG_HEADERS}" "${ARG_CLASSES}" "${WRAPPER_TEMPLATE}" MAKE_WRAPPER_BLOCK externals)  
  
  if(ARG_FILENAME)
    set(external_filename ${ARG_FILENAME})
  else()
    list(GET externals 0 external_filename)
  endif()
  
  if(ARG_EXTRA_SOURCE)
    message(WARNING "${ARG_EXTRA_SOURCE}")
    # set(EXTRA_SOURCE "${ARG_EXTRA_SOURCE}")    
    file(READ ${ARG_EXTRA_SOURCE} extra)
    string(APPEND MAKE_WRAPPER_BLOCK "${extra}")
    message(WARNING "${MAKE_WRAPPER_BLOCK}")
  endif()
  
  configure_file("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/client.cpp.in" generated_sources/${external_filename}.cpp)
  
  if(ARG_FILE_OUT)
    set(${ARG_FILE_OUT} generated_sources/${external_filename}.cpp PARENT_SCOPE)
  endif()
  
  if(ARG_EXTERNALS_OUT)
    set(${ARG_EXTERNALS_OUT} ${externals}  PARENT_SCOPE)
  endif()
endfunction()
