#define CATCH_CONFIG_MAIN

#include "SlicerTestHarness.hpp"
#include <algorithms/public/EnvelopeSegmentation.hpp>
#include <catch2/catch.hpp>
#include <data/FluidIndex.hpp>
#include <data/FluidTensor.hpp>
#include <Signals.hpp>
#include <TestUtils.hpp>
#include <algorithm>
#include <functional>
#include <string>
#include <vector>

namespace fluid {

using FastRampUp = StrongType<index, struct FastRampUpTag>;
using FastRampDown = StrongType<index, struct FastRampDownTag>;
using SlowRampUp = StrongType<index, struct SlowRampUpTag>;
using SlowRampDown = StrongType<index, struct SlowRampDownTag>;
using OnThreshold = StrongType<index, struct OnThresholdTag>;
using OffThreshold = StrongType<index, struct OffThresholdTag>;
using Floor = StrongType<index, struct FloorTag>;
using MinSliceLength = StrongType<index, struct MinSliceLengthTag>;
using HighPassFreq = StrongType<double, struct HighPassFreqTag>;
using Data = StrongType<FluidTensorView<const double, 1>, struct DataTag>;
using Expected = StrongType<std::vector<index>, struct ExpectedTag>;
using Margin = StrongType<index, struct MarginTag>;

struct TestParams
{
  FastRampUp     fastRampUp;
  FastRampDown   fastRampDown;
  SlowRampUp     slowRampUp;
  SlowRampDown   slowRampDown;
  OnThreshold    onThreshold;
  OffThreshold   offThreshold;
  Floor          floor;
  MinSliceLength minSliceLength;
  HighPassFreq   highPassFreq;
  Data           data;
  Expected       expected;
  Margin         margin;
};

std::vector<index> runTest(FluidTensorView<const double, 1> testSignal,
                           TestParams const&                p)
{

  double hiPassFreq = std::min<double>(p.highPassFreq / 44100, 0.5);

  auto algo = algorithm::EnvelopeSegmentation();
  algo.init(p.floor, hiPassFreq);

  std::vector<index> spikePositions;

  index i{0};

  for (auto&& x : testSignal)
  {
    if (algo.processSample(x, p.onThreshold, p.offThreshold, p.floor,
                           p.fastRampUp, p.slowRampUp, p.fastRampDown,
                           p.slowRampDown, hiPassFreq, p.minSliceLength) > 0)
    {
      spikePositions.push_back(i);
    }

    i++;
  }

  return spikePositions;
}


TEST_CASE("EnvSeg can be exactly precise with impulses", "[slicers][AmpSlice]")
{

  auto data = testsignals::monoImpulses();
  auto exp = testsignals::stereoImpulsePositions();

  auto params =
      TestParams{FastRampUp(10),     FastRampDown(2205), SlowRampUp(4410),
                 SlowRampDown(4410), OnThreshold(10),    OffThreshold(5),
                 Floor(-144),        MinSliceLength(2),  HighPassFreq(85),
                 Data(data),         Expected(exp),      Margin(1)};

  auto result = runTest(params.data(), params);
  REQUIRE_THAT(result, Catch::Equals(exp));
}

TEST_CASE("EnvSeg is predictable with sharp sine bursts", "[slicers][AmpSlice]")
{

  auto data = testsignals::sharpSines();
  auto exp = std::vector<index>{1001,  1455,  1493,  11028, 11412, 11450, 22053,
                                22399, 22437, 22475, 33078, 33462, 33500};

  auto params = TestParams{
      FastRampUp(5),    FastRampDown(50), SlowRampUp(220), SlowRampDown(220),
      OnThreshold(10),  OffThreshold(10), Floor(-60),      MinSliceLength(2),
      HighPassFreq(85), Data(data),       Expected(exp),   Margin(1)};

  auto result = runTest(params.data(), params);
  
  
  
  REQUIRE_THAT(result, Catch::Matchers::Approx(exp).margin(1));
}


TEST_CASE("EnvSeg schmitt triggering is predictable", "[slicers][AmpSlice]")
{

  auto data = testsignals::sharpSines();
  auto exp = std::vector<index>{1001, 11028, 22053, 33078};

  auto params = TestParams{
      FastRampUp(5),    FastRampDown(50), SlowRampUp(220), SlowRampDown(220),
      OnThreshold(10),  OffThreshold(5),  Floor(-60),      MinSliceLength(2),
      HighPassFreq(85), Data(data),       Expected(exp),   Margin(1)};

  auto result = runTest(params.data(), params);
//  REQUIRE_THAT(result, Catch::Equals(exp));
  REQUIRE_THAT(result, Catch::Matchers::Approx(exp).margin(1));
}

TEST_CASE("EnvSeg debouncing is predictable", "[slicers][AmpSlice]")
{

  auto data = testsignals::sharpSines();
  auto exp = std::vector<index>{1001, 11028, 22053, 33078};

  auto params = TestParams{
      FastRampUp(5),    FastRampDown(50), SlowRampUp(220), SlowRampDown(220),
      OnThreshold(10),  OffThreshold(10), Floor(-60),      MinSliceLength(800),
      HighPassFreq(85), Data(data),       Expected(exp),   Margin(1)};

  auto result = runTest(params.data(), params);
//  REQUIRE_THAT(result, Catch::Equals(exp));
REQUIRE_THAT(result, Catch::Matchers::Approx(exp).margin(1));
}

TEST_CASE("EnvSeg debouncing and Schmitt trigger together are predictable",
          "[slicers][AmpSlice]")
{

  auto data = testsignals::sharpSines();
  auto exp = std::vector<index>{1001, 22053};

  auto params =
      TestParams{FastRampUp(5),     FastRampDown(50),      SlowRampUp(220),
                 SlowRampDown(220), OnThreshold(10),       OffThreshold(5),
                 Floor(-60),        MinSliceLength(15000), HighPassFreq(85),
                 Data(data),        Expected(exp),         Margin(1)};

  auto result = runTest(params.data(), params);
//  REQUIRE_THAT(result, Catch::Equals(exp));
  REQUIRE_THAT(result, Catch::Matchers::Approx(exp).margin(1));
}

TEST_CASE("EnvSeg is predictable on real meaterial", "[slicers][AmpSlice]")
{

  auto data = testsignals::monoDrums();
  auto exp = std::vector<index>{1685,   38411,  51140,  69840,  88051,  114540,
                                151768, 176349, 202307, 220877, 239981, 252606,
                                259266, 276511, 283738, 289695, 296181, 302794,
                                326863, 353451, 372514, 390215, 417181};

  auto params =
      TestParams{FastRampUp(10),     FastRampDown(2205),   SlowRampUp(4410),
                 SlowRampDown(4410), OnThreshold(10),      OffThreshold(5),
                 Floor(-40),         MinSliceLength(4410), HighPassFreq(20),
                 Data(data),         Expected(exp),        Margin(1)};

  auto result = runTest(params.data(), params);
  REQUIRE_THAT(result, Catch::Equals(exp));
}


} // namespace fluid
