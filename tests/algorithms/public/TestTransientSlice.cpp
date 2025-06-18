#define APPROVALS_CATCH2_V3 

#include "SlicerTestHarness.hpp"
#include <flucoma/algorithms/public/TransientSegmentation.hpp>
#include <ApprovalTests.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <flucoma/data/FluidIndex.hpp>
#include <flucoma/data/FluidTensor.hpp>
#include <Signals.hpp>
#include <TestUtils.hpp>
#include <algorithm>
#include <functional>
#include <string>
#include <vector>

//use a subdir for approval test results 
auto directoryDisposer =
    ApprovalTests::Approvals::useApprovalsSubdirectory("approval_tests");

namespace fluid {

using Order = StrongType<index, struct OrderTag>;
using BlockSize = StrongType<index, struct BlockSizeTag>;
using Padding = StrongType<index, struct PaddingTag>;
using Skew = StrongType<double, struct SkewTag>;
using ThreshFwd = StrongType<double, struct ThreshFwdTag>;
using ThreshBack = StrongType<double, struct ThreshBackTag>;
using WindowSize = StrongType<index, struct WindowSizeTag>;
using ClumpLength = StrongType<index, struct ClumpLengthTag>;
using MinSliceLength = StrongType<index, struct MinSliceLengthTag>;

struct TestParams
{
  Order          order;
  BlockSize      blocksize;
  Padding        padding;
  Skew           skew;
  ThreshFwd      threshfwd;
  ThreshBack     threshback;
  WindowSize     windowsize;
  ClumpLength    clumplength;
  MinSliceLength minslicelength;
};

std::vector<index> runTest(FluidTensorView<const double, 1> testSignal,
                           TestParams const&                p)
{
  auto algo = algorithm::TransientSegmentation(p.order, p.blocksize, p.padding);
  algo.init(p.order, p.blocksize, p.padding);

  const double skew = pow(2, p.skew);

  const index maxWinIn = 2 * p.blocksize + p.padding;
  const index maxWinOut = maxWinIn;
  const index halfWindow = lrint(p.windowsize / 2);
  const index latency = p.padding + p.blocksize - p.order;

  algo.setDetectionParameters(skew, p.threshfwd, p.threshback, halfWindow,
                              p.clumplength, p.minslicelength);


  const index            hopSize = algo.hopSize();
  index                  nHops = std::ceil(testSignal.size() / hopSize);
  FluidTensor<double, 1> paddedInput(testSignal.size() + latency + hopSize);
  paddedInput(Slice(latency, testSignal.size())) <<= testSignal;

  FluidTensor<double, 1> output(hopSize);

  std::vector<index> spikePositions;

  index i{0};

  for (index i = 0; i < paddedInput.size(); i += hopSize)
  {
    algo.process(paddedInput(Slice(i, algo.inputSize())), output);

    auto it = std::find_if(output.begin(), output.end(),
                           [](double x) { return x > 0; });
    while (it != output.end())
    {
      spikePositions.push_back(std::distance(output.begin(), it) + i - latency);
      it = std::find_if(std::next(it), output.end(),
                        [](double x) { return x > 0; });
    }
  }

  return spikePositions;
}

TEST_CASE("TransientSlice is predictable on impulses",
          "[TransientSlice][slicers]")
{
  auto source = testsignals::monoImpulses();
  auto expected = testsignals::stereoImpulsePositions();

  auto params =
      TestParams{Order(20),      BlockSize(256),  Padding(128),
                 Skew(0),        ThreshFwd(2),    ThreshBack(1.1),
                 WindowSize(14), ClumpLength(25), MinSliceLength(1000)};        
  auto  matcher = Catch::Matchers::Approx(expected);
  index margin = 8;
  matcher.margin(margin);

  auto result = runTest(source, params);

  CHECK_THAT(result, matcher);
}


TEST_CASE("TransientSlice is predictable on sharp sine bursts",
          "[TransientSlice][slicers]")
{
  auto source = testsignals::sharpSines();
  auto expected = std::vector<index>{1000, 11025, 22050, 33075};

  auto params =
      TestParams{Order(20),      BlockSize(256),  Padding(128),
                 Skew(0),        ThreshFwd(2),    ThreshBack(1.1),
                 WindowSize(14), ClumpLength(25), MinSliceLength(1000)};

  auto  matcher = Catch::Matchers::Approx(expected);
  index margin = 8;
  matcher.margin(margin);

  auto result = runTest(source, params);

  CHECK_THAT(result, matcher);
}


// TEST_CASE("TransientSlice is predictable on real material",
//           "[TransientSlice][slicers]")
// {
//   auto source = testsignals::monoEurorackSynth();

//   auto params =
//       TestParams{Order(20),      BlockSize(256),  Padding(128),
//                  Skew(0),        ThreshFwd(2),    ThreshBack(1.1),
//                  WindowSize(14), ClumpLength(25), MinSliceLength(1000)};

//   auto result = runTest(source, params);
//   ApprovalTests::Approvals::verifyAll("slice points",result); 
// } 

// TEST_CASE("TransientSlice is predictable on real material with heavy settings",
//           "[TransientSlice][slicers]")
// {
//   auto source = testsignals::monoEurorackSynth();

//   auto params =
//       TestParams{Order(200),     BlockSize(2048), Padding(1024),
//                  Skew(1),        ThreshFwd(3),    ThreshBack(1),
//                  WindowSize(15), ClumpLength(30), MinSliceLength(4410)};


//   auto result = runTest(source(Slice(0, 220500)), params);
//   ApprovalTests::Approvals::verifyAll("slice points", result); 
// }


} // namespace fluid
