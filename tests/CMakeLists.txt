cmake_minimum_required (VERSION 3.11)

function (load_compile_tests test_name_stub sourcefile) 

  file(STRINGS ${sourcefile} FAILERS REGEX "#ifdef (FAIL_.+)")
  file(STRINGS ${sourcefile} CONFIMERS REGEX "#ifdef (CONFIRM_.+)")

  foreach(TESTLINE IN LISTS FAILERS)
    
    string(REGEX MATCH "FAIL_[^ \t\r\n]+" TESTNAME ${TESTLINE})
    message(STAUS "Fluid Tensor build failure tests: ${TESTNAME}") 
    add_compile_test(${TESTNAME} ${sourcefile} true)
  endforeach()

  foreach(TESTLINE IN LISTS CONFIMERS)
     
     string(REGEX MATCH "CONFIRM_[^ \t\r\n]+" TESTNAME ${TESTLINE}) 
     message(STAUS "Fluid Tensor build confirmation tests: ${TESTNAME}") 
     add_compile_test(${TESTNAME} ${sourcefile} false) 
  endforeach()

endfunction()


function (add_compile_test testname sourcefile should_fail)
  add_executable(${testname} "${sourcefile}")
  target_link_libraries(${testname} FLUID_DECOMPOSITION)
  set_target_properties(${testname} PROPERTIES EXCLUDE_FROM_ALL true
                                             EXCLUDE_FROM_DEFAULT_BUILD true
                                             CXX_STANDARD 14
                                             CXX_STANDARD_REQUIRED ON
                                             CXX_EXTENSIONS OFF
                                             POSITION_INDEPENDENT_CODE ON
                                         )                                             
  add_test(NAME ${testname}
   COMMAND ${CMAKE_COMMAND} --build "${CMAKE_BINARY_DIR}" --target ${testname} --config $<CONFIGURATION> -- -D${testname}
  )
  set_tests_properties(${testname} PROPERTIES WILL_FAIL ${should_fail})
endfunction()




function add_assert_fail_test(assert_fail testname sourcefile)
  add_executable(${testname} "${sourcefile}")
  target_link_libraries(${testname} PRIVATE FLUID_DECOMPOSITION)
  # set_target_properties(${testname} PROPERTIES CXX_STANDARD 14
  #                                              CXX_STANDARD_REQUIRED ON
  #                                             CXX_EXTENSIONS OFF
  #                                        )  
  add_test(NAME ${testname}
    COMMAND ${testname} 
  )
  set_tests_properties(${testname} PROPERTIES WILL_FAIL true)
endfunction()

Include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_SHALLOW    TRUE
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        devel
)
FetchContent_MakeAvailable(Catch2)

list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)

# add_executable(TestAR algorithms/util/TestAR.cpp algorithms/util/burg.cpp)
add_executable(TestFluidTensor data/TestFluidTensor.cpp)
target_link_libraries(TestFluidTensor PUBLIC Catch2::Catch2WithMain FLUID_DECOMPOSITION)

message(STATUS TestFluidTensor_TESTS)

include(CTest)
include(Catch)
catch_discover_tests(TestFluidTensor WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
# add_test(Test_FluidTensor COMMAND TestFluidTensor)

# add_compile_fail_test(NoBuildy data/TestFluidTensor_Compile.cpp)
load_compile_tests("FluidTensor Compilation Tests" data/compile_tests/TestFluidTensor_Compile.cpp) 
