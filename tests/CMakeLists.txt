cmake_minimum_required (VERSION 3.11)

add_library(AbortHandler OBJECT AbortHandler.cpp)
set_target_properties(AbortHandler PROPERTIES 
  EXCLUDE_FROM_ALL true
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON
)

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
                                         
  target_compile_definitions(${testname} PRIVATE ${testname})                                       
  add_test(NAME ${testname}
   COMMAND ${CMAKE_COMMAND} --build "${CMAKE_BINARY_DIR}" --target ${testname} --config $<CONFIGURATION> 
  )
  set_tests_properties(${testname} PROPERTIES WILL_FAIL ${should_fail})
endfunction()

function (add_assert_fail_test testname sourcedir)


try_compile(
  testname_BUILT "${CMAKE_BINARY_DIR}/tmp/${testname}" "${sourcedir}" FluidTensorDeathTests
  OUTPUT_VARIABLE whathappened
)

message(WARNING ${whathappened})

# add_subdirectory("${sourcedir}" EXCLUDE_FROM_ALL)
# include(${INSTALL_PREFIX}/.cmake)
if(testname_BUILT)
  # execute_process(

  # )
else()
  message(ERROR "Could not build assertion death tests in ${sourcedir}")
endif()
  add_custom_command(${testname}_BUILD)

  add_executable(${testname} "${sourcefile}")
  target_link_libraries(${testname} PRIVATE FLUID_DECOMPOSITION)
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
  GIT_TAG        v2.x
)
FetchContent_MakeAvailable(Catch2)

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
target_compile_definitions(TestUtils INTERFACE APPROVAL_TESTS_HIDE_DEPRECATED_CODE=1)
set_target_properties(TestUtils PROPERTIES 
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS  OFF
)


# #until approvalcpp is up to date with Catch2 changes 
# execute_process(
#   COMMAND ${CMAKE_COMMAND} -E create_symlink ${catch2_SOURCE_DIR}/src/catch2/catch_all.hpp ${catch2_SOURCE_DIR}/src/catch2/catch.hpp 
# )

list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/contrib)

# add_executable(TestAR algorithms/util/TestAR.cpp algorithms/util/burg.cpp)
add_executable(TestFluidTensor data/TestFluidTensor.cpp)
target_link_libraries(TestFluidTensor PUBLIC TestUtils)

add_executable(TestFluidTensorDeath data/death_tests/TestFluidTensorAsserts.cpp)
target_link_libraries(TestFluidTensorDeath PUBLIC 
  TestUtils  $<TARGET_OBJECTS:AbortHandler>   
)
set_target_properties(TestFluidTensorDeath PROPERTIES 
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS  OFF
)

add_executable(TestFluidTensorView data/TestFluidTensorView.cpp)
target_link_libraries(TestFluidTensorView PUBLIC TestUtils)
set_target_properties(TestFluidTensorView PROPERTIES 
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS  OFF
)


add_executable(TestFluidTensorSupport data/TestFluidTensorSupport.cpp)
target_link_libraries(TestFluidTensorSupport PUBLIC TestUtils)
set_target_properties(TestFluidTensorSupport PROPERTIES 
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS  OFF
)


add_executable(TestFluidDataSet data/TestFluidDataSet.cpp)
target_link_libraries(TestFluidDataSet PUBLIC TestUtils)
set_target_properties(TestFluidDataSet PROPERTIES 
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS  OFF
)

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

load_compile_tests("FluidTensor Compilation Tests" data/compile_tests/TestFluidTensor_Compile.cpp) 
