#undef NDEBUG

#define CATCH_CONFIG_MAIN 
#include <catch2/catch_all.hpp> 

// #include <catch2/catch_test_macros.hpp>
#include <flucoma/data/FluidTensor.hpp>

using fluid::FluidTensor;
using fluid::FluidTensorView;

TEST_CASE("Mismatched copy","[FluidTensor]"){
  const FluidTensor<int,2> x(3,3);
  FluidTensor<int,2> y(1,1);
  y <<= FluidTensorView<const int,2>(x);
}

TEST_CASE("Mismatched copy same overall size","[FluidTensor]"){
  const FluidTensor<int,2> x(3,3);
  FluidTensor<int,2> y(1,6); 
  y <<= FluidTensorView<const int,2>(x);
}

TEST_CASE("Mismatched converting copy","[FluidTensor]"){
  const FluidTensor<int,2> x(3,3); 
  FluidTensor<double,2> y(1,1); 
   y <<= FluidTensorView<const int,2>(x);
}

TEST_CASE("Mismatched converting copy same overall size","[FluidTensor]"){
  const FluidTensor<int,2> x(3,3); 
  FluidTensor<double,2> y(1,6); 
  y <<= FluidTensorView<const int,2>(x);
}

TEST_CASE("Row index out of bounds","[FluidTensor]"){
  const FluidTensor<int,2> x(1,1); 
  auto y = x.row(1);
}

TEST_CASE("Col index out of bounds","[FluidTensor]"){
  const FluidTensor<int,2> x(1,1); 
  auto y = x.col(1);
}

TEST_CASE("Element access out of bounds dim 0","[FluidTensor]"){
  const FluidTensor<int,2> x(1,1); 
  auto y = x(1,0);
}

TEST_CASE("Element access out of bounds dim 1","[FluidTensor]"){
  const FluidTensor<int,2> x(1,1); 
  auto y = x(0,1);
}
