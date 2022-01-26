#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
// #include <catch2/catch_test_macros.hpp>
#include <clients/common/BufferedProcess.hpp>
#include <clients/common/FluidContext.hpp>
#include <data/FluidTensor.hpp>
#include <CatchUtils.hpp>
#include <cmath> 
#include <algorithm>
#include <array>
#include <vector>

using fluid::EqualsRange;
using fluid::FluidTensor;
using fluid::FluidTensorView;
using fluid::Slice;
using fluid::client::BufferedProcess; 
using fluid::client::FluidContext;

TEST_CASE("BufferedProcess will reconstruct windowed input properly under COLA conditions","[BufferedProcess]"){
    
    BufferedProcess processor; 
    
    constexpr int hostSize = 64;
    FluidContext c;
    
    auto frameSize = GENERATE(32, 64, 256, 1024, 8192);
    
    auto hop = frameSize / 2; 
    
    processor.maxSize(frameSize, frameSize, 1,1 ); 
    processor.hostSize(hostSize); // sigh, FIXME
    
    FluidTensor<double,2> input(1,128 * frameSize);
    
    input(Slice(0),Slice(frameSize)).fill(1);
    
    FluidTensor<double,2> output(1,hostSize);
    FluidTensor<double, 2> window(1, frameSize);
    
    std::vector<double> expected(hostSize);
    std::vector<double> actual(hostSize);
    
    window.apply([frameSize, i=0](double& x) mutable { 
        x = 0.5 - 0.5 * cos((2 * M_PI * i++) / frameSize);
    }); //generate Hann window
    
    for(int i = frameSize; i < input.size() - hostSize;  i+=hostSize)
    {
      processor.push(input(Slice(0),Slice(i,hostSize)));
      
      //Hann windowing with overlap of 2 should be COLA
      processor.process(frameSize, frameSize,hop,c,
      [&window](FluidTensorView<double, 2> in, FluidTensorView<double, 2> out) {
        out = in;
        out.apply(window, [](double& x, double w) { x *= w; });
      });

      processor.pull(FluidTensorView<double,2>(actual.data(),0,1,hostSize));
      
      //we expect output to be input delayed by frameSize samples
      auto expectedSlice= input(Slice(0), Slice(i - frameSize, hostSize));
      std::copy(expectedSlice.begin(), expectedSlice.end(),expected.begin());
      
      auto matcher = Catch::Matchers::Approx(expected);
      double epsilon = 1e-12;
      matcher.epsilon(epsilon);
      
      CHECK_THAT(actual,matcher);
      
    }
}
