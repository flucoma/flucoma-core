#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
// #include <catch2/catch_test_macros.hpp>
#include <clients/common/FluidSource.hpp>
#include <data/FluidTensor.hpp>
#include <CatchUtils.hpp>
#include <algorithm>
#include <array>
#include <vector>

using fluid::EqualsRange;
using fluid::FluidSource;
using fluid::FluidTensor;
using fluid::FluidTensorView;
using fluid::Slice;

TEMPLATE_TEST_CASE(
    "FluidSource can provide different window and hop sizes with a known delay",
    "[FluidSource][frames]", int, double)
{

  constexpr int hostSize = 64;
  constexpr int maxFrameSize = 1024;

  FluidSource<TestType> framer(maxFrameSize, 1);
  framer.setHostBufferSize(hostSize);
  framer.reset(1); // sigh, FIXME

  std::array<TestType, 2 * maxFrameSize> data;
  std::array<TestType, maxFrameSize>     output;

  std::iota(data.begin(), data.end(), 0);

  // run the test with each of these frame sizes
  auto frameSize = GENERATE(32, 43, 64, 96, 128, 512);


  // and for each frame size above, we test with these hops
  auto overlap = GENERATE(4, 3, 2, 1);
  int  hop = frameSize / overlap;

  FluidTensor<TestType, 1> expected(data.size() + frameSize);
  expected(Slice(frameSize)) <<=
      FluidTensorView<TestType, 1>(data.data(), 0, data.size());

  for (int i = 0, j = 0, k = 0; i < data.size() - hostSize; i += hostSize)
  {
    auto input = FluidTensorView<TestType, 2>{data.data(), i, 1, hostSize};
    framer.push(input);
    auto outputView =
        FluidTensorView<TestType, 2>{output.data(), 0, 1, frameSize};

    for (; j < hostSize; j += hop, k += hop)
    {
      framer.pull(outputView, j);
      CHECK_THAT(outputView, EqualsRange(expected(Slice(k, frameSize))));
    }

    j = j < hostSize ? j : j - hostSize;
  }
}
