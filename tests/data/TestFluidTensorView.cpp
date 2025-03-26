#define CATCH_CONFIG_MAIN 
#include <catch2/catch.hpp> 
// #include <catch2/catch_test_macros.hpp>
// #include <catch2/matchers/catch_matchers_templated.hpp>
#include <flucoma/data/FluidTensor.hpp> 
#include <flucoma/data/FluidMeta.hpp> 
#include <CatchUtils.hpp> 


#include <array>
#include <vector>
#include <algorithm> 

using fluid::FluidTensor; 
using fluid::FluidTensorView; 
using fluid::Slice;
using fluid::FluidTensorSlice; 

using fluid::EqualsRange; 


TEST_CASE("FluidTensorView can be constructed from a pointer and a slice","[FliudTensorView]"){ 
    std::array<int,6> x{0,1,2,3,4,5}; 

    SECTION("1D"){
        FluidTensorSlice<1> desc{0, {6}}; 
        FluidTensorView<int,1> y(desc,x.data()); 

        CHECK(y.data() == x.data()); 
        CHECK(y.size() == x.size()); 
        CHECK(y.descriptor() == desc); 
    }

    SECTION("2D")
    {
        FluidTensorSlice<2> desc{0,{2,3}}; 
        FluidTensorView<int, 2>y(desc,x.data()); 

        CHECK(y.data() == x.data()); 
        CHECK(y.size() == x.size()); 
        CHECK(y.descriptor() == desc); 
        CHECK(y.rows() == 2); 
        CHECK(y.cols() == 3); 
        REQUIRE_THAT(y,EqualsRange(x)); 
        REQUIRE_THAT(y.row(0),EqualsRange(std::array<int,3>{0,1,2})); 
        REQUIRE_THAT(y.row(1),EqualsRange(std::array<int,3>{3,4,5})); 
        REQUIRE_THAT(y.col(0),EqualsRange(std::array<int,2>{0,3})); 
        REQUIRE_THAT(y.col(1),EqualsRange(std::array<int,2>{1,4})); 
        REQUIRE_THAT(y.col(2),EqualsRange(std::array<int,2>{2,5})); 
    }

    SECTION("1D with offset"){
        FluidTensorSlice<1> desc{1, {4}}; 
        FluidTensorView<int,1> y(desc,x.data()); 

        CHECK(y.data() == x.data() + 1); 
        CHECK(y.size() == 4); 
        REQUIRE_THAT(y,EqualsRange(std::array<int,4>{1,2,3,4}));     
    }

     SECTION("2D with Offset")
    {
        FluidTensorSlice<2> desc{2,{2,2}}; 
        FluidTensorView<int, 2>y(desc,x.data()); 
        
        CHECK(y.data() == x.data() + 2); 
        CHECK(y.size() == 4); 
        CHECK(y.descriptor() == desc); 
        CHECK(y.rows() == 2); 
        CHECK(y.cols() == 2); 
        REQUIRE_THAT(y,EqualsRange(std::array<int,4>{2,3,4,5})); 
        REQUIRE_THAT(y.row(0),EqualsRange(std::array<int,2>{2,3})); 
        REQUIRE_THAT(y.row(1),EqualsRange(std::array<int,2>{4,5})); 
        REQUIRE_THAT(y.col(0),EqualsRange(std::array<int,2>{2,4})); 
        REQUIRE_THAT(y.col(1),EqualsRange(std::array<int,2>{3,5}));     
    }
}

TEST_CASE("FluidTensorView can have its order increased similar to np.newaxis","[FluidTensorView]"){

    auto data = std::array<int,3>{1,2,3}; 
    auto x = FluidTensorView<int,1>{data.data(),0,3}; 
    auto y = FluidTensorView<int,2>{x}; 

    CHECK(y.rows() == 1); 
    CHECK(y.cols() == 3);     
    REQUIRE_THAT(y.row(0),EqualsRange(data)); 
    REQUIRE_THAT(y.col(0),EqualsRange(std::array<int,1>{1})); 
    REQUIRE_THAT(y.col(1),EqualsRange(std::array<int,1>{2})); 
    REQUIRE_THAT(y.col(2),EqualsRange(std::array<int,1>{3})); 
}


TEST_CASE("FluidTensorView copy construnction is shallow","[FluidTensorView]")
{
    auto data = std::array<int,3>{1,2,3}; 
    auto x = FluidTensorView<int,1>{data.data(),0,3}; 
    auto y = FluidTensorView<int,1>{x}; 

    CHECK(y.data() == x.data()); 
    CHECK(y.descriptor() == x.descriptor());     
}

TEST_CASE("FluidTensorView copy assignment is deep","[FluidTensorView]")
{
    auto data = std::array<int,3>{1,2,3}; 
    auto data2 = std::array<int,3>{-1,-1,-1}; 
    auto x = FluidTensorView<int,1>{data.data(),0,3}; 
    auto y = FluidTensorView<int,1>{data2.data(),0,3}; 

    y <<= x; 

    CHECK(y.data() != x.data()); 
    CHECK(y.descriptor() == x.descriptor());    
    REQUIRE_THAT(data2,EqualsRange(data)); 
}

TEST_CASE("FluidTensorView can be reset to a new pointer","[FluidTensorView]"){
    auto data = std::array<int,3>{1,2,3}; 
    auto data2 = std::array<int,5>{-1,-1,4,5,6}; 
    auto x = FluidTensorView<int,1>{data.data(),0,3}; 

    x.reset(data2.data(),2,data2.size() - 2); 

    CHECK(x.size() == data2.size() - 2); 
    CHECK(x.descriptor().start == 2); 
    REQUIRE_THAT(x,EqualsRange(std::array<int,3>{4,5,6}));     
}

TEST_CASE("FluidTensorView can be accessed by index","[FluidTensorView]"){
    auto data = std::array<int,6>{1,2,3,4,5,6}; 
    
    SECTION("1D"){
        auto x = FluidTensorView<int,1>{data.data(),0,6}; 

        for(fluid::index i = 0; i < data.size(); ++i)
        {
            CHECK(x(i) == data[i]); 
            CHECK(x[i] == data[i]); 
        }
    }

    SECTION("2D"){
        auto x = FluidTensorView<int,2>{data.data(),0,2,3}; 

        for(fluid::index i = 0; i < x.rows(); ++i)
        {
           for(fluid::index j = 0; j < x.cols(); ++j)
           {
               CHECK(x(i,j) == data[(i * x.cols()) + j]);
           }
        } 
    }
}

TEST_CASE("FluidTensorView can be accessed by slice","[FluidTensorView]"){

    std::array<int, 10> data{0,1,2,3,4,5,6,7,8,9}; 
    const auto x =  FluidTensorView<int,2>(data.data(),0,2,5); 

    SECTION("Cols"){     
      for(int i = 0; i < 5; ++i)
      {
        const auto y = x(Slice(0),i);  
        CHECK(y.size() == 2); 
        REQUIRE_THAT(y,EqualsRange(std::array<int,2>{i, i + 5}));     
      }    
    }

    SECTION("Rows"){
      //rows    
      for(int i = 0; i < 2; ++i)
      {
        const auto y = x(i,Slice(0));  
        CHECK(y.size() == 5); 
        REQUIRE_THAT(y,EqualsRange(
            std::array<int,5>{(i * 5), 
                             (i * 5) + 1, 
                             (i * 5) + 2, 
                             (i * 5) + 3,
                             (i * 5) + 4}
        )); 
      }
    }

    SECTION("Rectangles"){
        const auto y = x(Slice(0,2),Slice(1,2));//both rows, 2 cols offset 1       
        CHECK(y.size() == 4); 
        REQUIRE_THAT(y,EqualsRange(std::array<int,4>{1,2,6,7})); 
    }
}

TEST_CASE("FluidTensorView can be transposed","[FluidTensorView]")
{
    std::array<int,10> data{0,1,2,3,4,5,6,7,8,9}; 
    auto x =  FluidTensorView<int,2>(data.data(),0,2,5); 
    auto xT = x.transpose(); 

    CHECK(xT.rows() == x.cols()); 
    CHECK(xT.cols() == x.rows()); 

    for(fluid::index i = 0; i < x.rows(); ++i) 
    {
        REQUIRE_THAT(xT.col(i),EqualsRange(x.row(i))); 
    }

    for(fluid::index i = 0; i < x.cols(); ++i) 
    {
        REQUIRE_THAT(xT.row(i),EqualsRange(x.col(i))); 
    }
    
    auto xTT = xT.transpose(); 

    CHECK(xTT.rows() == x.rows()); 
    CHECK(xTT.cols() == x.cols()); 

    for(fluid::index i = 0; i < x.rows(); ++i) 
    {
        REQUIRE_THAT(xTT.row(i),EqualsRange(x.row(i))); 
    }

    for(fluid::index i = 0; i < x.cols(); ++i) 
    {
        REQUIRE_THAT(xTT.col(i),EqualsRange(x.col(i))); 
    }
}
