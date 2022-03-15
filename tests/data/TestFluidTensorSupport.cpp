#define CATCH_CONFIG_MAIN 
#include <catch2/catch.hpp> 
// #include <catch2/catch_test_macros.hpp>
// #include <catch2/matchers/catch_matchers_templated.hpp>
#include <data/FluidIndex.hpp>
#include <data/FluidTensor_Support.hpp> 
#include <data/FluidMeta.hpp> 
#include <CatchUtils.hpp> 

#include <data/FluidTensor.hpp> 

#include <array>
#include <algorithm> 

using fluid::Slice;
using fluid::FluidTensorSlice; 
using fluid::impl::SliceIterator; 
using fluid::EqualsRange; 

TEST_CASE("FluidTensorSlice can be  default constructed, copied, moved","[FluidTensorSupport]")
{
    auto x = FluidTensorSlice<1>(); 

    CHECK(x.start == 0); 
    CHECK(x.size == 0); 
    CHECK(x.transposed == false); 
    REQUIRE_THAT(x.extents,EqualsRange(std::array<fluid::index,1>{0})); 
    REQUIRE_THAT(x.strides,EqualsRange(std::array<fluid::index,1>{0})); 

    auto y = x; //copy 

    CHECK(y.start == x.start); 
    CHECK(y.size == x.size); 
    CHECK(y.transposed == x.transposed); 
    REQUIRE_THAT(y.extents,EqualsRange(x.extents)); 
    REQUIRE_THAT(y.strides,EqualsRange(x.strides)); 

    auto z = std::move(y); //move 

    CHECK(z.start == x.start); 
    CHECK(z.size == x.size); 
    CHECK(z.transposed == x.transposed); 
    REQUIRE_THAT(z.extents,EqualsRange(x.extents)); 
    REQUIRE_THAT(z.strides,EqualsRange(x.strides)); 

    auto x2 = FluidTensorSlice<2>(); //2D

    CHECK(x2.start == 0); 
    CHECK(x2.size == 0); 
    REQUIRE_THAT(x2.extents,EqualsRange(std::array<fluid::index,2>{0,0})); 
    REQUIRE_THAT(x2.strides,EqualsRange(std::array<fluid::index,2>{0,0})); 
}

TEST_CASE("FluidTensorSlice can be constructed from dimension lists","[FluidTensorSupport]"){

    std::array<fluid::index,2> extents{8,13}; 

    FluidTensorSlice<2> x{0,extents}; 

    CHECK(x.start == 0); 
    CHECK(x.size ==  8 * 13); 
    REQUIRE_THAT(x.extents,EqualsRange(extents)); 
    REQUIRE_THAT(x.strides,EqualsRange(std::array<fluid::index,2>{extents[1], 1})); 

    FluidTensorSlice<2> y{8,13}; 

    CHECK(y.start == 0); 
    CHECK(y.size ==  x.size); 
    REQUIRE_THAT(x.extents,EqualsRange(extents)); 
    REQUIRE_THAT(x.strides,EqualsRange(std::array<fluid::index,2>{extents[1], 1})); 
}

TEST_CASE("FluidTensorSlice can construct sub-slices","[FluidTensorSupport]"){
    FluidTensorSlice<2> x{8,13}; 

    SECTION("Row"){ 
        FluidTensorSlice<2> y(x,0,Slice(0)); 
        CHECK(y.start == x.start); 
        CHECK(y.size == x.extents[1]); 
        REQUIRE_THAT(y.extents,EqualsRange(std::array<fluid::index,2>{1,13})); 
        REQUIRE_THAT(y.strides,EqualsRange(std::array<fluid::index,2>{13,1})); 
    }

    SECTION("Col"){ 
        FluidTensorSlice<2> y(x,Slice(0),0); 
        CHECK(y.start == x.start); 
        CHECK(y.size == x.extents[0]); 
        REQUIRE_THAT(y.extents,EqualsRange(std::array<fluid::index,2>{8,1})); 
        REQUIRE_THAT(y.strides,EqualsRange(std::array<fluid::index,2>{13,1})); 
    }

    SECTION("Rect"){ 
        FluidTensorSlice<2> y(x,Slice(1),Slice(1)); 
        CHECK(y.start == 1 + x.strides[0]); 
        CHECK(y.size == 7 * 12); 
        REQUIRE_THAT(y.extents,EqualsRange(std::array<fluid::index,2>{x.extents[0] - 1, x.extents[1] - 1})); 
        REQUIRE_THAT(y.strides,EqualsRange(std::array<fluid::index,2>{13,1})); 
    }
}

TEST_CASE("FluidTensorSlice operator() maps indices back to flat layout","[FluidTensorSuppport]"){
    std::array<int, 54> data; 
    std::iota(data.begin(),data.end(),0); 

    SECTION("1D"){
        FluidTensorSlice<1> x{54}; 
        for(const auto i:data)
            CHECK(data[x(i)] == data[i]); 
    }

    SECTION("2D"){
        FluidTensorSlice<2> x{18,3}; 

        for(int i = 0; i < 18; i++)
            for(int j; j < 3; j++)
               CHECK(data[x(i,j)] == data[(i * x.extents[1]) + j]);  
    }

    SECTION("3D"){
        FluidTensorSlice<3> x{2,9,3}; 

        for(int i = 0; i < 2; i++)
            for(int j = 0; j < 9; j++)
                for(int k = 0; k < 3; k++ )
                    CHECK(data[x(i,j,k)] == data[(i * x.extents[2] * x.extents[1]) + (j * x.extents[2]) + (k)]);  
    }
}

TEST_CASE("FluidTensorSlice can be grown","[FluidTensorSupport]"){
        FluidTensorSlice<2> x{5,3}; 
        
        x.grow(0,5); 
        CHECK(x.size == 10 * 3); 
        REQUIRE_THAT(x.extents,EqualsRange(std::array<fluid::index,2>{10,3})); 
        REQUIRE_THAT(x.strides,EqualsRange(std::array<fluid::index,2>{3,1})); 
        
        x.grow(0,-5); 
        CHECK(x.size == 5 * 3); 
        REQUIRE_THAT(x.extents,EqualsRange(std::array<fluid::index,2>{5,3})); 
        REQUIRE_THAT(x.strides,EqualsRange(std::array<fluid::index,2>{3,1})); 
        
        x.grow(1,5); 
        CHECK(x.size == 5 * 8); 
        REQUIRE_THAT(x.extents,EqualsRange(std::array<fluid::index,2>{5,8})); 
        REQUIRE_THAT(x.strides,EqualsRange(std::array<fluid::index,2>{8,1})); 
}

TEST_CASE("FluidTensorSlice can produce a transposed copy","[FluidTensorSupport]"){
    FluidTensorSlice<2> x{5,3};         
    auto y = x.transpose(); 

    CHECK(y.transposed == true); 
    REQUIRE_THAT(y.extents,EqualsRange(std::array<fluid::index,2>{3,5})); 
    REQUIRE_THAT(y.strides,EqualsRange(std::array<fluid::index,2>{1,3})); 
}

TEST_CASE("SliceIterator does strided iteration","[FluidTensorSupport]"){
    FluidTensorSlice<2> s{5,3}; 
    std::array<int,15> data; 
    std::iota(data.begin(),data.end(),0); 
    

    using Iterator = fluid::impl::SliceIterator<int,2>; 

    auto i = Iterator(s,data.data()); 
    auto end = Iterator(s,data.data(),true); 
    
    SECTION("Deference"){
        CHECK(std::equal(i,end,data.begin())); 
    }
    
    SECTION("Prefix and postfix increment are same"){
        auto i2 = Iterator(s,data.data()); 
        for(; i != end; ++i, i2++)
            CHECK(*i == *i2); 
    }

    SECTION("Striding over non-contiguous data"){
        FluidTensorSlice<2> col(s,Slice(0),0); 
        std::array<int, 5>  colData{0,3,6,9,12}; 
        auto colIterator = Iterator(col,data.data()); 
        auto end = Iterator(col,data.data(), true); 
        REQUIRE(std::equal(colIterator,end,colData.begin())); 
    }
}



TEST_CASE("SliceIterator end() does sensible things","[FluidTensorSupport]"){
  
  std::array<int,10>  data{0,1,2,3,4,5,6,7,8,9}; 
  fluid::FluidTensorView<int,2> view{data.data(),0,1,10};
  auto size = GENERATE(0,1,2,3,4,5,6,7,8,9); 
  
  using Iterator = fluid::impl::SliceIterator<int,1>; 
  FluidTensorSlice<1> s(size);
  
  auto colIterator = Iterator(s,data.data()); 
  auto end = Iterator(s,data.data(), true); 
  size_t distance=std::distance(colIterator,end); 
  
  CHECK(size == distance);
  
  auto slice = view(Slice(0),Slice(0,size));
  distance=std::distance(slice.begin(),slice.end());
  CHECK(size == distance);
  
  
  
  
}
