cmake_minimum_required (VERSION 3.11)

add_library(TestSignals STATIC Signals.cpp)
target_include_directories(TestSignals PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(TestSignals PRIVATE FLUID_DECOMPOSITION HISSTools_AudioFile)
target_compile_features(TestSignals PUBLIC cxx_std_14)
