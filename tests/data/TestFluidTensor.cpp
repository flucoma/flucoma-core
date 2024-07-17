#define CATCH_CONFIG_MAIN 
#include <catch2/catch.hpp> 
// #include <catch2/catch_test_macros.hpp>
#include <data/FluidTensor.hpp> 
#include <data/FluidMeta.hpp> 

#include <array>
#include <vector>
#include <algorithm> 

using fluid::FluidTensor; 
using fluid::FluidTensorView; 
using fluid::Slice;

  
TEST_CASE("FluidTensor can be created from a list of dimenions","[FluidTensor]"){
  
  SECTION("1D creation reports correct sizes"){
    const FluidTensor<int,1> x(3); 
    
    REQUIRE(x.size() == 3);     
    REQUIRE(x.rows() == 3); 
    // REQUIRE(x.cols() == 1); 
    REQUIRE(std::distance(x.begin(),x.end()) == x.size()); 
  }
  SECTION("2D creation reports correct sizes"){
    const FluidTensor<int,2> x(3,2); 
    
    REQUIRE(x.size() == (3 * 2));     
    REQUIRE(x.rows() == 3); 
    REQUIRE(x.cols() == 2); 
    REQUIRE(std::distance(x.begin(),x.end()) == x.size()); 
  }
  SECTION("3D creation reports correct sizes"){
    const FluidTensor<int,3> x(3,2,5); 
    
    REQUIRE(x.size() == (3 * 2 * 5));     
    REQUIRE(x.rows() == 3); 
    REQUIRE(x.cols() == 2); 
    REQUIRE(x.descriptor().extents[2] == 5); 
    REQUIRE(std::distance(x.begin(),x.end()) == x.size()); 
  }
}

TEST_CASE("FluidTensor can be initialized from initializer lists","[FluidTensor]"){

  const std::array<int, 5> y{0,1,2,3,4}; 
  std::array<int, 5> y1;  
  std::transform(y.begin(), y.end(),y1.begin(),[](int x){ return x + 5; }); 
  
  const std::array<int, 2> c{0,5}; 
  const std::array<int, 2> c1{1,6};
  const std::array<int, 2> c2{2,7}; 
  const std::array<int, 2> c3{3,8}; 
  const std::array<int, 2> c4{4,9}; 

  SECTION("1D initialization"){
    const FluidTensor<int, 1> x{0,
                          1,
                          2,
                          3,
                          4}; 
    const std::array<int, 5> y{0,1,2,3,4}; 
    REQUIRE(x.rows() == 5); 
  //   REQUIRE(x.cols() == 1); 
    REQUIRE(std::distance(x.begin(),x.end()) == x.size()); 
    REQUIRE(std::equal(x.begin(),x.end(), y.begin()));
  }  

  SECTION("2D initialization"){
    //@todo I found a bug: you can enter ragged lists and it compiles :-( 
    // my incliination is to kill this feature 
    const FluidTensor<int, 2> x{{0,1,2,3,4},
                          {5,6,7,8,9}}; 
    REQUIRE(x.size() == (2 * 5));     
    REQUIRE(x.rows() == 2); 
    REQUIRE(x.cols() == 5); 
    REQUIRE(std::distance(x.begin(),x.end()) == x.size()); 
    REQUIRE(std::equal(x.row(0).begin(),x.row(0).end(), y.begin()));
    REQUIRE(std::equal(x.row(1).begin(),x.row(1).end(), y1.begin()));

    REQUIRE(std::equal(x.col(0).begin(),x.col(0).end(), c.begin()));
    REQUIRE(std::equal(x.col(1).begin(),x.col(1).end(), c1.begin()));
    REQUIRE(std::equal(x.col(2).begin(),x.col(2).end(), c2.begin()));
    REQUIRE(std::equal(x.col(3).begin(),x.col(3).end(), c3.begin()));
    REQUIRE(std::equal(x.col(4).begin(),x.col(4).end(), c4.begin()));
  }  
}

TEST_CASE("FluidTensor data can be accessed by index and by slice","[FluidTensor]"){

    const FluidTensor<int, 2> x{{0,1,2,3,4},{5,6,7,8,9}}; 

    SECTION("Index Access"){
      for(int i = 0; i < 2; ++i)
        for(int j = 0; j < 5; ++j)
          CHECK(x(i,j) == (i  * 5) + j);
    }

    SECTION("Slice Access"){
      //cols
      for(int i = 0; i < 5; ++i)
      {
        const auto y = x(Slice(0),i);  
        const std::array<int, 2> z = {i, i + 5}; 
        CHECK(y.size() == 2); 
        CHECK(std::equal(y.begin(),y.end(),z.begin())); 
      }

      //rows    
      for(int i = 0; i < 2; ++i)
      {
        const auto y = x(i,Slice(0));  
        const std::array<int, 5> z = {(i * 5), (i * 5) + 1, (i * 5) + 2, (i * 5) + 3, (i * 5) + 4}; 
        CHECK(y.size() == 5); 
        CHECK(std::equal(y.begin(),y.end(),z.begin())); 
      }
    }

    //rectangles 
    {
      const auto y = x(Slice(0,2),Slice(1,2));//both rows, 2 cols offset 1       
      const std::array<int, 4> z = {1,2,6,7}; 
      CHECK(y.size() == 4); 
      CHECK(std::equal(y.begin(),y.end(),z.begin())); 
    }

}

TEST_CASE("FluidTensor can be copied","[FluidTensor]"){
  const  FluidTensor<int, 2> x{{0,1,2,3,4},{5,6,7,8,9}};  
  
  SECTION("Copy construct") {
      const FluidTensor<int, 2>  y(x); 
      CHECK(y.size() == x.size()); 
      CHECK(y.rows() == x.rows()); 
      CHECK(y.cols() == x.cols()); 
      CHECK(y.descriptor() == x.descriptor()); 
      CHECK(std::equal(y.begin(),y.end(),x.begin())); 
  }
   
  SECTION("Copy assign") {
      FluidTensor<int, 2>  y;
      y = x; 
      CHECK(y.size() == x.size()); 
      CHECK(y.rows() == x.rows()); 
      CHECK(y.cols() == x.cols()); 
      CHECK(y.descriptor() == x.descriptor()); 
      CHECK(std::equal(y.begin(),y.end(),x.begin())); 
  }

  SECTION("Copy algorithm") {
      FluidTensor<int, 2> y(2,5);
      std::copy(x.begin(),x.end(),y.begin());  
      CHECK(y.size() == x.size()); 
      CHECK(y.rows() == x.rows()); 
      CHECK(y.cols() == x.cols()); 
      CHECK(y.descriptor() == x.descriptor()); 
      CHECK(std::equal(y.begin(),y.end(),x.begin())); 
  }

  SECTION("Copy conversion") {
      const FluidTensor<double, 2> y(x);  
      CHECK(y.size() == x.size()); 
      CHECK(y.rows() == x.rows()); 
      CHECK(y.cols() == x.cols()); 
      CHECK(y.descriptor() == x.descriptor()); 
      CHECK(std::equal(y.begin(),y.end(),x.begin())); 
  }
}

TEST_CASE("FluidTensor can be moved","[FluidTensor]"){
  FluidTensor<int, 2> x{{0,1,2,3,4},{5,6,7,8,9}};  
  SECTION("Move construct") {
      const FluidTensor<int, 2>  y(FluidTensor<int,2>{{0,1,2,3,4},{5,6,7,8,9}}); 
      CHECK(y.size() == x.size()); 
      CHECK(y.rows() == x.rows()); 
      CHECK(y.cols() == x.cols()); 
      CHECK(y.descriptor() == x.descriptor()); 
      CHECK(std::equal(y.begin(),y.end(),x.begin())); 
  }
   
  SECTION("Move assign") {
      FluidTensor<int, 2>  y;
      y = FluidTensor<int,2>{{0,1,2,3,4},{5,6,7,8,9}}; 
      CHECK(y.size() == x.size()); 
      CHECK(y.rows() == x.rows()); 
      CHECK(y.cols() == x.cols()); 
      CHECK(y.descriptor() == x.descriptor()); 
      CHECK(std::equal(y.begin(),y.end(),x.begin())); 
  }
}

TEST_CASE("FluidTensor can be resized","[FluidTensor]"){
  SECTION("resize gives correct dimensions")
  {
    FluidTensor<int,2> x(1,1); 
    x.resize(3,4); 
    CHECK(x.size() == (3 * 4));     
    CHECK(x.rows() == 3); 
    CHECK(x.cols() == 4); 
  }

  SECTION("resize dim correct dimensions")
  {
    FluidTensor<int,2> x(1,1); 
    x.resizeDim(0,2); 
    x.resizeDim(1,3); 
    CHECK(x.size() == (3 * 4));     
    CHECK(x.rows() == 3); 
    CHECK(x.cols() == 4); 
  }

  SECTION("delete row preserves data")
  {
    FluidTensor<int,2> x{
      {0,1,2},{3,4,5},{6,7,8}
    }; 
    x.deleteRow(1); 
    std::array<int,6> y{0,1,2,6,7,8}; 
    CHECK(x.size() == 6); 
    CHECK(x.rows() == 2); 
    CHECK(x.cols() == 3); 
    REQUIRE(std::equal(x.begin(),x.end(),y.begin())); 
  }
}

TEST_CASE("const FluidTensor returns View(const T)","[FluidTensor]"){
  SECTION("const rows and cols"){
    const FluidTensor<int,2> x{
      {0,1,2},{3,4,5},{6,7,8}
    }; 
    CHECK(std::is_same<decltype(x.row(0)),FluidTensorView<const int, 1>>());
    CHECK(std::is_same<decltype(x.col(0)),FluidTensorView<const int, 1>>());
    CHECK(std::is_same<decltype(x.transpose()),const FluidTensorView<const int, 2>>());
    CHECK(std::is_same<decltype(x(Slice(0),Slice(0))),const FluidTensorView<const int, 2>>());
    CHECK(std::is_same<decltype(x(Slice(0),0)), const FluidTensorView<const int, 2>>());
    CHECK(std::is_same<decltype(x(0,0)),const int&>());
  }
}
