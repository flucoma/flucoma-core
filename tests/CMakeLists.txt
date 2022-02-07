cmake_minimum_required (VERSION 3.11)

###### Utils 
add_subdirectory(test_signals)

##### Assert Death Testing 

# a handler to catch sigabrt for us on death tests
add_library(AbortHandler OBJECT AbortHandler.cpp)
set_target_properties(AbortHandler PROPERTIES 
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS  OFF
)

# function (add_assert_fail_test testname sourcedir)
#   try_compile(
#     testname_BUILT "${CMAKE_BINARY_DIR}/tmp/${testname}" "${sourcedir}" FluidTensorDeathTests
#     OUTPUT_VARIABLE whathappened
#   )
#   message(WARNING ${whathappened})
# 
#   if(NOT testname_BUILT)
#     message(ERROR "Could not build assertion death tests in ${sourcedir}")
#   endif()
# 
#   add_custom_command(${testname}_BUILD)
#   add_executable(${testname} "${sourcefile}")
#   target_link_libraries(${testname} PRIVATE FLUID_DECOMPOSITION)
# 
#   add_test(NAME ${testname}
#     COMMAND ${testname} 
#   )
#   set_tests_properties(${testname} PROPERTIES WILL_FAIL true)
# endfunction()

###### compilation tests 
function (add_compile_tests test_name_stub sourcefile) 

  file(STRINGS ${sourcefile} FAILERS REGEX "#ifdef (FAIL_.+)")
  file(STRINGS ${sourcefile} CONFIMERS REGEX "#ifdef (CONFIRM_.+)")
  
  foreach(TESTLINE IN LISTS FAILERS)    
    string(REGEX MATCH "FAIL_[^ \t\r\n]+" TESTNAME ${TESTLINE})
    # message(STATUS "Fluid Tensor build failure tests: ${TESTNAME}") 
    _add_one_compile_test(${TESTNAME} ${sourcefile} true)
  endforeach()

  foreach(TESTLINE IN LISTS CONFIMERS)     
     string(REGEX MATCH "CONFIRM_[^ \t\r\n]+" TESTNAME ${TESTLINE}) 
     # message(STATUS "Fluid Tensor build confirmation tests: ${TESTNAME}") 
     _add_one_compile_test(${TESTNAME} ${sourcefile} false) 
  endforeach()

endfunction()

function (_add_one_compile_test testname sourcefile should_fail)
  add_executable(${testname} "${sourcefile}")
  target_link_libraries(${testname} FLUID_DECOMPOSITION)
  set_target_properties(${testname} PROPERTIES 
      EXCLUDE_FROM_ALL true
      EXCLUDE_FROM_DEFAULT_BUILD true
      CXX_STANDARD 14
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
      POSITION_INDEPENDENT_CODE ON
  )                                               
  target_compile_definitions(${testname} PRIVATE ${testname})
  #test consists of running compiler on source 
  add_test(NAME ${testname}
   COMMAND ${CMAKE_COMMAND} --build "${CMAKE_BINARY_DIR}" --target ${testname} --config $<CONFIGURATION> 
  )
  set_tests_properties(${testname} PROPERTIES WILL_FAIL ${should_fail})
endfunction()

#*********************************************************************
# main body 

Include(FetchContent)

# get catch 
FetchContent_Declare(
  Catch2
  GIT_SHALLOW    TRUE
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v2.x
)
FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/contrib)

# get approvaltests.cpp 
FetchContent_Declare( 
  ApprovalCPP 
  GIT_SHALLOW    TRUE
  GIT_REPOSITORY https://github.com/approvals/ApprovalTests.cpp.git
  GIT_TAG        master
)
FetchContent_MakeAvailable(ApprovalCPP)

add_library(TestUtils INTERFACE)
target_include_directories(TestUtils  INTERFACE include)
target_link_libraries(TestUtils INTERFACE
  Catch2::Catch2
  FLUID_DECOMPOSITION
  ApprovalTests::ApprovalTests
)
target_compile_definitions(TestUtils INTERFACE 
  APPROVAL_TESTS_HIDE_DEPRECATED_CODE=1
)
target_compile_features(TestUtils INTERFACE cxx_std_14)

function(add_test_executable target_name source_file)
  add_executable(${target_name} ${source_file})
  target_link_libraries(${target_name} PUBLIC TestUtils)
  set_target_properties(${target_name} PROPERTIES 
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS  OFF
  )
endfunction()

add_test_executable(TestFluidTensor data/TestFluidTensor.cpp)
add_test_executable(TestFluidTensorDeath data/death_tests/TestFluidTensorAsserts.cpp)
target_link_libraries(TestFluidTensorDeath PRIVATE   
    $<TARGET_OBJECTS:AbortHandler>   
)

add_test_executable(TestFluidTensorView data/TestFluidTensorView.cpp)
add_test_executable(TestFluidTensorSupport data/TestFluidTensorSupport.cpp)
add_test_executable(TestFluidDataSet data/TestFluidDataSet.cpp)
add_test_executable(TestFluidSource clients/common/TestFluidSource.cpp)
add_test_executable(TestFluidSink clients/common/TestFluidSink.cpp)
add_test_executable(TestBufferedProcess clients/common/TestBufferedProcess.cpp)

add_test_executable(TestNoveltySeg 
  algorithms/public/TestNoveltySegmentation.cpp
)
add_test_executable(TestOnsetSeg algorithms/public/TestOnsetSegmentation.cpp)
add_test_executable(TestEnvelopeSeg algorithms/public/TestEnvelopeSegmentation.cpp)

add_test_executable(TestEnvelopeGate algorithms/public/TestEnvelopeGate.cpp)

add_test_executable(TestTransientSlice algorithms/public/TestTransientSlice.cpp)


target_link_libraries(TestNoveltySeg PRIVATE TestSignals)
target_link_libraries(TestOnsetSeg PRIVATE TestSignals)
target_link_libraries(TestEnvelopeSeg PRIVATE TestSignals)
target_link_libraries(TestEnvelopeGate PRIVATE TestSignals)
target_link_libraries(TestTransientSlice PRIVATE TestSignals)

include(CTest)
include(Catch)

catch_discover_tests(TestFluidTensor WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")

catch_discover_tests(TestFluidTensorDeath 
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
  PROPERTIES WILL_FAIL true
)

catch_discover_tests(TestFluidTensorView WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
catch_discover_tests(TestFluidTensorSupport WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
catch_discover_tests(TestFluidDataSet WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")

catch_discover_tests(TestNoveltySeg WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
catch_discover_tests(TestOnsetSeg WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
catch_discover_tests(TestEnvelopeSeg WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
catch_discover_tests(TestEnvelopeGate WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
catch_discover_tests(TestTransientSlice WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")

catch_discover_tests(TestFluidSource WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
catch_discover_tests(TestFluidSink WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
catch_discover_tests(TestBufferedProcess WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")

add_compile_tests("FluidTensor Compilation Tests" data/compile_tests/TestFluidTensor_Compile.cpp) 
