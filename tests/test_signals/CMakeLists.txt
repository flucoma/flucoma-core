cmake_minimum_required (VERSION 3.11)

get_filename_component(FLUCOMA_CORE_AUDIO 
  "${CMAKE_CURRENT_SOURCE_DIR}/../../Resources/AudioFiles" ABSOLUTE
)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Signals.cpp.in ${CMAKE_BINARY_DIR}/Signals.cpp @ONLY)

add_library(TestSignals STATIC ${CMAKE_BINARY_DIR}/Signals.cpp)
target_include_directories(TestSignals PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(TestSignals PRIVATE 
  FLUID_DECOMPOSITION
)
target_compile_features(TestSignals PUBLIC cxx_std_14)
