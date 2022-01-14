cmake_minimum_required(VERSION 3.11)
project(FluidTensorDeathTests LANGUAGES CXX)

add_executable(
    TestFluidTensorAsserts ${CMAKE_CURRENT_SOURCE_DIR}/TestFluidTensorAsserts.cpp
)    
target_link_libraries(TestFluidTensorAsserts FLUID_DECOMPOSITION Catch2WithMain)

# include(GNUInstallDirs)
install(TARGETS 
TestFluidTensorAsserts
EXPORT fluidTensorDeathTests
)
install(EXPORT fluidTensorDeathTests
    DESTINATION ${CMAKE_SOURCE_DIR}
    NAMESPACE FluidTensorDeathTests::
    FILE FluidTensorDeathTests.cmake
)

