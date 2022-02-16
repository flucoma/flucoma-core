#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
// #include <catch2/catch_test_macros.hpp>
#include <clients/common/FluidSink.hpp>
#include <data/FluidTensor.hpp>
#include <CatchUtils.hpp>
#include <algorithm>
#include <array>
#include <vector>

using fluid::EqualsRange;
using fluid::FluidSink;
using fluid::FluidTensor;
using fluid::FluidTensorView;
using fluid::Slice;


TEMPLATE_TEST_CASE("FluidSink can overlap-add a varitety of window sizes and hops",
    "[FluidSink][frames]", int, double)
{

  constexpr int hostSize = 64;
  constexpr int maxFrameSize = 1024;

  FluidSink<TestType> olaBuffer(maxFrameSize, 1);
  olaBuffer.setHostBufferSize(hostSize);
  olaBuffer.reset(1); // sigh, FIXME

  std::array<TestType, 2 * maxFrameSize> data;
  data.fill(1); 
  
  
  std::array<TestType, maxFrameSize>     emptyFrame;

  emptyFrame.fill(0);
//  std::iota(data.begin(), data.end(), 0);

  // run the test with each of these frame sizes
  auto frameSize = GENERATE(32, 43, 64, 96, 128, 512);

  FluidTensor<TestType, 1> expected(data.size());
  expected.fill(0);
  FluidTensor<TestType, 2> output(1, hostSize);


  // and for each frame size above, we test with these overlaps
  auto overlap = GENERATE(1,2,3,4);
  
  int hop = frameSize / overlap;
  
  //simulate ouptut data by building buffer for whole span
  for(int i = 0; i < expected.size();i+=hop)
  {
    int chunkSize = std::min<int>(frameSize, expected.size() - i);
    expected(Slice(i,chunkSize)).apply([](TestType& x){
      x += 1;
    });
  }
  

  for (int i = 0, j = 0, k = 0; i < data.size() - hostSize; i += hostSize)
  {
    
    for (; j < hostSize; j += hop, k += 1)
    {
      auto input = FluidTensorView<TestType, 2>{ data.data(), i, 1, frameSize };
      olaBuffer.push(input, j);
    }
   
    j = j < hostSize ? j : j - hostSize;
    olaBuffer.pull(FluidTensorView<TestType,2>(output));
    CHECK_THAT(output,EqualsRange(expected(Slice(i,hostSize))));
  }
}
