include_guard()

find_package(Git REQUIRED)

execute_process( 
  COMMAND ${GIT_EXECUTABLE} describe --abbrev=0
  RESULT_VARIABLE result
  OUTPUT_VARIABLE FLUID_VERSION_TAG
  OUTPUT_STRIP_TRAILING_WHITESPACE  
)

if(result)
  message(FATAL_ERROR "Failed to get version string from Git")
endif()

configure_file("${CMAKE_CURRENT_LIST_DIR}/FluidVersion.hpp.in" "${CMAKE_CURRENT_LIST_DIR}/FluidVersion.hpp" @ONLY)
set(FLUID_VERSION_PATH ${CMAKE_CURRENT_LIST_DIR})  
