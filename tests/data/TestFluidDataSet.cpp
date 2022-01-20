#define APPROVALS_CATCH
#define CATCH_CONFIG_MAIN 
// #include <catch2/catch.hpp> 
#include <ApprovalTests.hpp>

// #include <catch2/catch_test_macros.hpp>
#include <data/FluidTensor.hpp> 
#include <data/FluidMeta.hpp> 
#include <data/FluidDataSet.hpp> 
#include <CatchUtils.hpp> 

#include <array>
#include <vector>
#include <algorithm> 
#include <string> 

using DataSet = fluid::FluidDataSet<std::string, int, 1>; 
using fluid::FluidTensor; 
using fluid::FluidTensorView; 
using fluid::Slice; 
using fluid::EqualsRange; 

//use a subdir for approval test results 
auto directoryDisposer =
    ApprovalTests::Approvals::useApprovalsSubdirectory("approval_tests");

TEST_CASE("FluidDataSet can be default constructed","[FluidDataSet]")
{
    DataSet d; 
    CHECK(d.initialized() == false); 
    CHECK(d.size() == 0); 
    CHECK(d.dims() == 0); 
    CHECK(d.getIds().size() == 0); 
    CHECK(d.getData().size() == 0); 
}

TEST_CASE("FluidDataSet can be constructed from data","[FluidDataSet]")
{
    FluidTensor<int, 2> points{{0,1,2,3,4},{5,6,7,8,9}}; 
    FluidTensor<std::string,1> labels{"zero","one"}; 

    SECTION("Non-converting")
    {
        DataSet d(labels, points); 
        CHECK(d.initialized() == true); 
        CHECK(d.size() == 2); 
        CHECK(d.dims() == 5); 
        CHECK(d.getIds().size() == 2); 
        CHECK(d.getData().size() == 10);
    } 

    SECTION("Converting")
    {
        FluidTensor<double, 2> float_points(points); 
        
        DataSet d(FluidTensorView<std::string,1>{labels}, 
                  FluidTensorView<const double, 2>{float_points}); 
        CHECK(d.initialized() == true); 
        CHECK(d.size() == 2); 
        CHECK(d.dims() == 5); 
        CHECK(d.getIds().size() == 2); 
        CHECK(d.getData().size() == 10);
        REQUIRE_THAT(d.getData(),EqualsRange(points)); 
    } 

}

TEST_CASE("FluidDataSet can have points added","[FluidDataSet]")
{
    FluidTensor<int, 2> points{{0,1,2,3,4},{5,6,7,8,9}}; 
    FluidTensor<std::string,1> labels{"zero","one"}; 
    DataSet d(5); 

    CHECK(d.add(labels(0),points.row(0)) == true); 

    CHECK(d.size() == 1);
    CHECK(d.initialized() == true); 
    CHECK(d.dims() == 5); 
    REQUIRE_THAT(d.getData(),EqualsRange(points.row(0))); 
    REQUIRE_THAT(d.getIds(),EqualsRange(labels(Slice(0,1))));   

    CHECK(d.add(labels.row(1),points.row(1)) == true); 

    CHECK(d.size() == 2);
    CHECK(d.initialized() == true); 
    CHECK(d.dims() == 5); 
    REQUIRE_THAT(d.getData(),EqualsRange(points)); 
    REQUIRE_THAT(d.getIds(),EqualsRange(labels));   
}

TEST_CASE("FluidDataSet can have points retreived","[FluidDataSet]")
{
    FluidTensor<int, 2> points{{0,1,2,3,4},{5,6,7,8,9}}; 
    FluidTensor<std::string,1> labels{"zero","one"}; 
    DataSet d(5); 

    d.add(labels(0),points.row(0)); 
    d.add(labels(1),points.row(1)); 
    FluidTensor<int, 1> output{-1,-1,-1,-1,-1}; 

    CHECK(d.get(labels(0),output) == true); 
    REQUIRE_THAT(output,EqualsRange(points.row(0))); 

    CHECK(d.get(labels(1),output) == true); 
    REQUIRE_THAT(output,EqualsRange(points.row(1))); 

    CHECK(d.get("two",output) == false); 
    //output should be unchanged
    REQUIRE_THAT(output,EqualsRange(points.row(1))); 
}

TEST_CASE("FluidDataSet can have points updated","[FluidDataSet]")
{
    FluidTensor<int, 2> points{{0,1,2,3,4},{5,6,7,8,9}}; 
    FluidTensor<std::string,1> labels{"zero","one"}; 
    DataSet d(5); 

    d.add(labels(0),points.row(0)); 
    CHECK(d.update(labels(0),points.row(1)) == true); 
    
    CHECK(d.update(labels(1),points.row(1)) == false); 

    FluidTensor<int, 1> output{-1,-1,-1,-1,-1}; 
    d.get(labels(0),output); 
    
    REQUIRE_THAT(output,EqualsRange(points.row(1))); 
}

TEST_CASE("FluidDataSet can have points removed","[FluidDataSet]")
{
    FluidTensor<int, 2> points{{0,1,2,3,4},{5,6,7,8,9}}; 
    FluidTensor<std::string,1> labels{"zero","one"}; 
    DataSet d(5); 
    FluidTensor<int, 1> output{-1,-1,-1,-1,-1}; 

    d.add(labels(0),points.row(0)); 
    CHECK(d.remove(labels(0)) == true); 
    CHECK(d.remove(labels(0)) == false); 
    CHECK(d.initialized() == false); 
    CHECK(d.size() == 0); 
    CHECK(d.dims() == 5); //harrumph
    CHECK(d.getIds().size() == 0); 
    CHECK(d.getData().size() == 0); 
    CHECK(d.update(labels(0),points.row(0)) == false); 
    CHECK(d.get(labels(0),output) == false); 

    CHECK(d.add(labels(0),points.row(0)) == true); 
    CHECK(d.add(labels(1),points.row(1)) == true); 
    CHECK(d.remove(labels(0)) == true); 
    CHECK(d.size() == 1); 
    CHECK(d.dims() == 5); //harrumph
    CHECK(d.getIds().size() == 1); 
    CHECK(d.getData().size() == 5); 
    CHECK(d.update(labels(0),points.row(0)) == false); 
    CHECK(d.get(labels(0),output) == false);
}

TEST_CASE("FluidDataSet prints consistent summaries for approval","[FluidDataSet]")
{
    using namespace ApprovalTests; 

    SECTION("small")
    {
        FluidTensor<int, 2> points{{0,1,2,3,4},{5,6,7,8,9}}; 
        FluidTensor<std::string,1> labels{"zero","one"}; 

        DataSet d(labels, points); 

        Approvals::verify(d.print());
    }

    SECTION("bigger"){
        FluidTensor<int, 2> points(100,100); 
        std::iota(points.begin(), points.end(),0); 
        FluidTensor<std::string,1> labels(100); 
        std::transform(labels.begin(),labels.end(),labels.begin(), 
        [n=0](std::string s)mutable {
            return std::to_string(n); 
        }); 

        DataSet d(labels, points); 

        Approvals::verify(d.print()); 
    }

}
