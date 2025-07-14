#define CATCH_CONFIG_MAIN 
#include <catch2/catch_all.hpp> 
// #include <catch2/catch_test_macros.hpp>
#include <flucoma/data/FluidTensor.hpp> 
#include <flucoma/data/FluidMeta.hpp> 

#include <array>
#include <vector>
#include <algorithm> 

using fluid::FluidTensor; 
using fluid::FluidTensorView; 
using fluid::Slice;

  
// TEST_CASE("FluidTensor can be created from a list of dimenions","[FluidTensor]"){
  
//   SECTION("1D creation reports correct sizes"){
//     const FluidTensor<int,1> x(3); 
    
//     REQUIRE(x.size() == 3);     
//     REQUIRE(x.rows() == 3); 
//     // REQUIRE(x.cols() == 1); 
//     REQUIRE(std::distance(x.begin(),x.end()) == x.size()); 
//   }
//   SECTION("2D creation reports correct sizes"){
//     const FluidTensor<int,2> x(3,2); 
    
//     REQUIRE(x.size() == (3 * 2));     
//     REQUIRE(x.rows() == 3); 
//     REQUIRE(x.cols() == 2); 
//     REQUIRE(std::distance(x.begin(),x.end()) == x.size()); 
//   }
//   SECTION("3D creation reports correct sizes"){
//     const FluidTensor<int,3> x(3,2,5); 
    
//     REQUIRE(x.size() == (3 * 2 * 5));     
//     REQUIRE(x.rows() == 3); 
//     REQUIRE(x.cols() == 2); 
//     REQUIRE(x.descriptor().extents[2] == 5); 
//     REQUIRE(std::distance(x.begin(),x.end()) == x.size()); 
//   }
// }

// TEST_CASE("FluidTensor can be initialized from initializer lists","[FluidTensor]"){

//   const std::array<int, 5> y{0,1,2,3,4}; 
//   std::array<int, 5> y1;  
//   std::transform(y.begin(), y.end(),y1.begin(),[](int x){ return x + 5; }); 
  
//   const std::array<int, 2> c{0,5}; 
//   const std::array<int, 2> c1{1,6};
//   const std::array<int, 2> c2{2,7}; 
//   const std::array<int, 2> c3{3,8}; 
//   const std::array<int, 2> c4{4,9}; 

//   SECTION("1D initialization"){
//     const FluidTensor<int, 1> x{0,
//                           1,
//                           2,
//                           3,
//                           4}; 
//     const std::array<int, 5> y{0,1,2,3,4}; 
//     REQUIRE(x.rows() == 5); 
//     REQUIRE(std::distance(x.begin(),x.end()) == x.size()); 
//     REQUIRE_THAT(x,Catch::Matchers::RangeEquals(y)); 
//   }  

//   SECTION("2D initialization"){
//     //@todo I found a bug: you can enter ragged lists and it compiles :-( 
//     // my incliination is to kill this feature 
//     const FluidTensor<int, 2> x{{0,1,2,3,4},
//                           {5,6,7,8,9}}; 
//     REQUIRE(x.size() == (2 * 5));     
//     REQUIRE(x.rows() == 2); 
//     REQUIRE(x.cols() == 5); 
//     REQUIRE(std::distance(x.begin(),x.end()) == x.size()); 
//     REQUIRE_THAT(x.row(0),Catch::Matchers::RangeEquals(y)); 
//     REQUIRE_THAT(x.row(1),Catch::Matchers::RangeEquals(y1)); 

//     REQUIRE_THAT(x.col(0),Catch::Matchers::RangeEquals(c)); 
//     REQUIRE_THAT(x.col(1),Catch::Matchers::RangeEquals(c1)); 
//     REQUIRE_THAT(x.col(2),Catch::Matchers::RangeEquals(c2)); 
//     REQUIRE_THAT(x.col(3),Catch::Matchers::RangeEquals(c3)); 
//     REQUIRE_THAT(x.col(4),Catch::Matchers::RangeEquals(c4)); 
//   }  
// }

// TEST_CASE("FluidTensor data can be accessed by index and by slice","[FluidTensor]"){

//     const FluidTensor<int, 2> x{{0,1,2,3,4},{5,6,7,8,9}}; 

//     SECTION("Index Access"){
//       for(int i = 0; i < 2; ++i)
//         for(int j = 0; j < 5; ++j)
//           CHECK(x(i,j) == (i  * 5) + j);
//     }

//     SECTION("Slice Access"){
//       //cols
//       for(int i = 0; i < 5; ++i)
//       {
//         const auto y = x(Slice(0),i);  
//         const std::array<int, 2> z = {i, i + 5}; 
//         CHECK(y.size() == 2); 
//         CHECK_THAT(y, Catch::Matchers::RangeEquals(z)); 
//       }

//       //rows    
//       for(int i = 0; i < 2; ++i)
//       {
//         const auto y = x(i,Slice(0));  
//         const std::array<int, 5> z = {(i * 5), (i * 5) + 1, (i * 5) + 2, (i * 5) + 3, (i * 5) + 4}; 
//         CHECK(y.size() == 5); 
//         CHECK_THAT(y, Catch::Matchers::RangeEquals(z)); 
//       }
//     }

//     //rectangles 
//     {
//       const auto y = x(Slice(0,2),Slice(1,2));//both rows, 2 cols offset 1       
//       const std::array<int, 4> z = {1,2,6,7}; 
//       CHECK(y.size() == 4); 
//       CHECK_THAT(y, Catch::Matchers::RangeEquals(z)); 
//     }

// }

TEST_CASE("FluidTensor can be copied","[FluidTensor]"){
  const  FluidTensor<int, 2> x{{0,1,2,3,4},{5,6,7,8,9}};  
  
  SECTION("Copy construct") {
      const FluidTensor<int, 2>  y(x); 
      CHECK(y.size() == x.size()); 
      CHECK(y.rows() == x.rows()); 
      CHECK(y.cols() == x.cols()); 
      CHECK(y.descriptor() == x.descriptor()); 
      CHECK_THAT(y, Catch::Matchers::RangeEquals(x)); 
  }
   
  SECTION("Copy assign") {
      FluidTensor<int, 2>  y;
      y = x; 
      CHECK(y.size() == x.size()); 
      CHECK(y.rows() == x.rows()); 
      CHECK(y.cols() == x.cols()); 
      CHECK(y.descriptor() == x.descriptor()); 
      CHECK_THAT(y, Catch::Matchers::RangeEquals(x)); 
  }

  SECTION("Copy algorithm") {
      FluidTensor<int, 2> y(2,5);
      std::copy(x.begin(),x.end(),y.begin());  
      CHECK(y.size() == x.size()); 
      CHECK(y.rows() == x.rows()); 
      CHECK(y.cols() == x.cols()); 
      CHECK(y.descriptor() == x.descriptor()); 
      CHECK_THAT(y, Catch::Matchers::RangeEquals(x)); 
  }

  SECTION("Copy conversion") {
      const FluidTensor<double, 2> y(x);  
      CHECK(y.size() == x.size()); 
      CHECK(y.rows() == x.rows()); 
      CHECK(y.cols() == x.cols()); 
      CHECK(y.descriptor() == x.descriptor()); 
      CHECK_THAT(y, Catch::Matchers::RangeEquals(x)); 
  }
}

// TEST_CASE("FluidTensor can be moved","[FluidTensor]"){
//   FluidTensor<int, 2> x{{0,1,2,3,4},{5,6,7,8,9}};  
//   SECTION("Move construct") {
//       const FluidTensor<int, 2>  y(FluidTensor<int,2>{{0,1,2,3,4},{5,6,7,8,9}}); 
//       CHECK(y.size() == x.size()); 
//       CHECK(y.rows() == x.rows()); 
//       CHECK(y.cols() == x.cols()); 
//       CHECK(y.descriptor() == x.descriptor()); 
//       CHECK_THAT(y, Catch::Matchers::RangeEquals(x)); 
//   }
   
//   SECTION("Move assign") {
//       FluidTensor<int, 2>  y;
//       y = FluidTensor<int,2>{{0,1,2,3,4},{5,6,7,8,9}}; 
//       CHECK(y.size() == x.size()); 
//       CHECK(y.rows() == x.rows()); 
//       CHECK(y.cols() == x.cols()); 
//       CHECK(y.descriptor() == x.descriptor()); 
//       CHECK_THAT(y, Catch::Matchers::RangeEquals(x)); 
//   }
// }

// TEST_CASE("FluidTensor can be resized","[FluidTensor]"){
//   SECTION("resize gives correct dimensions")
//   {
//     FluidTensor<int,2> x(1,1); 
//     x.resize(3,4); 
//     CHECK(x.size() == (3 * 4));     
//     CHECK(x.rows() == 3); 
//     CHECK(x.cols() == 4); 
//   }

//   SECTION("resize dim correct dimensions")
//   {
//     FluidTensor<int,2> x(1,1); 
//     x.resizeDim(0,2); 
//     x.resizeDim(1,3); 
//     CHECK(x.size() == (3 * 4));     
//     CHECK(x.rows() == 3); 
//     CHECK(x.cols() == 4); 
//   }

//   SECTION("delete row preserves data")
//   {
//     FluidTensor<int,2> x{
//       {0,1,2},{3,4,5},{6,7,8}
//     }; 
//     x.deleteRow(1); 
//     std::array<int,6> y{0,1,2,6,7,8}; 
//     CHECK(x.size() == 6); 
//     CHECK(x.rows() == 2); 
//     CHECK(x.cols() == 3); 
//     REQUIRE_THAT(x, Catch::Matchers::RangeEquals(y)); 
//   }
// }

// TEST_CASE("const FluidTensor returns View(const T)","[FluidTensor]"){
//   SECTION("const rows and cols"){
//     const FluidTensor<int,2> x{
//       {0,1,2},{3,4,5},{6,7,8}
//     }; 
//     CHECK(std::is_same<decltype(x.row(0)),FluidTensorView<const int, 1>>());
//     CHECK(std::is_same<decltype(x.col(0)),FluidTensorView<const int, 1>>());
//     CHECK(std::is_same<decltype(x.transpose()),const FluidTensorView<const int, 2>>());
//     CHECK(std::is_same<decltype(x(Slice(0),Slice(0))),const FluidTensorView<const int, 2>>());
//     CHECK(std::is_same<decltype(x(Slice(0),0)), const FluidTensorView<const int, 2>>());
//     CHECK(std::is_same<decltype(x(0,0)),const int&>());
//   }
// }
