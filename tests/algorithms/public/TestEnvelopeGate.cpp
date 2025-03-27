#define CATCH_CONFIG_MAIN

#include "SlicerTestHarness.hpp"
#include <flucoma/algorithms/public/EnvelopeGate.hpp>
#include <catch2/catch.hpp>
#include <flucoma/data/FluidIndex.hpp>
#include <flucoma/data/FluidTensor.hpp>
#include <Signals.hpp>
#include <TestUtils.hpp>
#include <algorithm>
#include <functional>
#include <string>
#include <vector>

namespace fluid {

using RampUp = StrongType<index, struct RampUpTag>;
using RampDown = StrongType<index, struct RampDownTag>;
using OnThreshold = StrongType<index, struct OnThresholdTag>;
using OffThreshold = StrongType<index, struct OffThresholdTag>;
using MinLengthAbove = StrongType<index, struct MinLengthAboveTag>;
using MinLengthBelow = StrongType<index, struct MinSilenceLengthTag>;
using MinSliceLength = StrongType<index, struct MinSliceLengthTag>;
using MinSilenceLength = StrongType<index, struct MinSilenceLengthTag>;
using HighPassFreq = StrongType<double, struct HighPassFreqTag>;
using LookAhead = StrongType<double, struct LookAheadTag>;
using LookBack = StrongType<double, struct LookBackTag>;
using Data = StrongType<FluidTensorView<const double, 1>, struct DataTag>;
using Expected = StrongType<std::vector<index>, struct ExpectedTag>;
using Margin = StrongType<index, struct MarginTag>;


struct TestParams
{
  RampUp           rampUp;
  RampDown         rampDown;
  OnThreshold      onThreshold;
  OffThreshold     offThreshold;
  MinLengthAbove   minLengthAbove;
  MinLengthBelow   minLengthBelow;
  MinSliceLength   minSliceLength;
  MinSilenceLength minSilenceLength;
  LookAhead        lookahead;
  LookBack         lookback;
  HighPassFreq     highPassFreq;
};

std::pair<std::vector<index>, std::vector<index>>
runTest(FluidTensorView<const double, 1> testSignal, TestParams const& p)
{

  double hiPassFreq = std::min<double>(p.highPassFreq / 44100, 0.5);

  auto algo = algorithm::EnvelopeGate(88200);
  algo.init(p.onThreshold, p.offThreshold, hiPassFreq, p.minLengthAbove,
            p.lookback, p.minLengthBelow, p.lookahead);


  index latency =
      std::max<index>(p.minLengthAbove + p.lookback,
                      std::max<index>(p.minLengthBelow, p.lookahead));

  std::vector<index> onsetPositions;
  std::vector<index> offsetPositions;

  index i{0};

  bool state = false;

  for (auto&& x : testSignal)
  {


    double response = algo.processSample(x, p.onThreshold, p.offThreshold,
                                         p.rampUp, p.rampDown, hiPassFreq,
                                         p.minSliceLength, p.minSilenceLength);
    if (response > 0 && !state)
    {
      onsetPositions.push_back(i - latency);
      state = true;
    }
    else if (response == 0 && state)
    {
      offsetPositions.push_back(i - latency);
      state = false;
    }
    i++;
  }

  return {onsetPositions, offsetPositions};
}


TEST_CASE("EnvelopeGate is almost exact with impulses", "[AmpGate][slicers]")
{
  auto sig = testsignals::monoImpulses();
  auto expectedOnsets = testsignals::stereoImpulsePositions();
  auto expectedOffsets = testsignals::stereoImpulsePositions();

  std::transform(expectedOffsets.begin(), expectedOffsets.end(),
                 expectedOffsets.begin(), [](index x) { return x + 596; });

  auto params =
      TestParams{RampUp(1),         RampDown(10),        OnThreshold(-30),
                 OffThreshold(-90), MinLengthAbove(1),   MinLengthBelow(1),
                 MinSliceLength(1), MinSilenceLength(1), LookAhead(0),
                 LookBack(0),       HighPassFreq(85)};

  auto result = runTest(sig, params);

  REQUIRE(result.first.size() == result.second.size());

  auto  matcherOn = Catch::Matchers::Approx(expectedOnsets);
  index margin = 2;
  matcherOn.margin(margin);

  CHECK_THAT(result.first, matcherOn);

  auto matcherOff = Catch::Matchers::Approx(expectedOffsets);
  matcherOff.margin(margin);

  CHECK_THAT(result.second, matcherOff);
}

TEST_CASE("EnvelopeGate is predictable with smooth sine bursts",
          "[AmpGate][slicers]")
{
  auto sig = testsignals::smoothSine();
  auto expectedOnsets = std::vector<index>{
      1876,  1942,  2009,  2076,  2144,  2212,  2279,  2347,  2415,  2483,
      2551,  2619,  2687,  19431, 19501, 19571, 19641, 19711, 19781, 19851,
      19921, 19991, 20062, 20133, 20205, 23926, 23992, 24059, 24126, 24194,
      24262, 24329, 24397, 24465, 24533, 24601, 24669, 24737, 41481, 41551,
      41621, 41691, 41761, 41831, 41901, 41971, 42041, 42112, 42183, 42255};

  auto expectedOffsets = std::vector<index>{
      1890,  1965,  2039,  2113,  2185,  2257,  2329,  2400,  2471,  2542,
      2613,  2684,  19430, 19497, 19564, 19631, 19698, 19764, 19831, 19897,
      19963, 20028, 20092, 20156, 20219, 23940, 24015, 24089, 24163, 24235,
      24307, 24379, 24450, 24521, 24592, 24663, 24734, 41480, 41547, 41614,
      41681, 41748, 41814, 41881, 41947, 42013, 42078, 42142, 42206, 42269};


  auto params =
      TestParams{RampUp(5),         RampDown(25),        OnThreshold(-12),
                 OffThreshold(-12), MinLengthAbove(1),   MinLengthBelow(1),
                 MinSliceLength(1), MinSilenceLength(1), LookAhead(0),
                 LookBack(0),       HighPassFreq(85)};


  auto result = runTest(sig, params);

  REQUIRE(result.first.size() == result.second.size());

  auto  matcherOn = Catch::Matchers::Approx(expectedOnsets);
  index margin = 1;
  matcherOn.margin(margin);

  CHECK_THAT(result.first, matcherOn);

  auto matcherOff = Catch::Matchers::Approx(expectedOffsets);
  matcherOff.margin(margin);

  CHECK_THAT(result.second, matcherOff);
}

TEST_CASE("EnvelopeGate is predictable with smooth sine bursts and hysteresis",
          "[AmpGate][slicers]")
{
  auto sig = testsignals::smoothSine();
  auto expectedOnsets = std::vector<index>{1878, 23928};

  auto expectedOffsets = std::vector<index>{20462, 42512};

  auto params =
      TestParams{RampUp(5),         RampDown(25),        OnThreshold(-12),
                 OffThreshold(-16), MinLengthAbove(1),   MinLengthBelow(1),
                 MinSliceLength(1), MinSilenceLength(1), LookAhead(0),
                 LookBack(0),       HighPassFreq(85)};


  auto result = runTest(sig, params);

  REQUIRE(result.first.size() == result.second.size());

  auto  matcherOn = Catch::Matchers::Approx(expectedOnsets);
  index margin = 1;
  matcherOn.margin(margin);

  CHECK_THAT(result.first, matcherOn);

  auto matcherOff = Catch::Matchers::Approx(expectedOffsets);
  matcherOff.margin(margin);

  CHECK_THAT(result.second, matcherOff);
}

TEST_CASE("EnvelopeGate is predictable with smooth sine bursts and debouncing",
          "[AmpGate][slicers]")
{
  auto sig = testsignals::smoothSine();
  auto expectedOnsets =
      std::vector<index>{1876, 2347, 19431, 19921, 23926, 24397, 41481, 41971};

  auto expectedOffsets =
      std::vector<index>{2329, 19430, 19897, 20362, 24379, 41480, 41947, 42412};

  auto params =
      TestParams{RampUp(5),           RampDown(25),        OnThreshold(-12),
                 OffThreshold(-12),   MinLengthAbove(1),   MinLengthBelow(1),
                 MinSliceLength(441), MinSilenceLength(1), LookAhead(0),
                 LookBack(0),         HighPassFreq(85)};


  auto result = runTest(sig, params);

  REQUIRE(result.first.size() == result.second.size());

  auto  matcherOn = Catch::Matchers::Approx(expectedOnsets);
  index margin = 1;
  matcherOn.margin(margin);

  CHECK_THAT(result.first, matcherOn);

  auto matcherOff = Catch::Matchers::Approx(expectedOffsets);
  matcherOff.margin(margin);

  CHECK_THAT(result.second, matcherOff);
}

TEST_CASE(
    "EnvelopeGate is predictable with smooth sine bursts and gap debouncing",
    "[AmpGate][slicers]")
{
  auto sig = testsignals::smoothSine();
  auto expectedOnsets =
      std::vector<index>{1876, 2347, 2841, 19871, 23926, 24397, 24891, 41921};

  auto expectedOffsets =
      std::vector<index>{1890, 2400, 19430, 19897, 23940, 24450, 41480, 41947};

  auto params =
      TestParams{RampUp(5),         RampDown(25),          OnThreshold(-12),
                 OffThreshold(-12), MinLengthAbove(1),     MinLengthBelow(1),
                 MinSliceLength(1), MinSilenceLength(441), LookAhead(0),
                 LookBack(0),       HighPassFreq(85)};


  auto result = runTest(sig, params);

  REQUIRE(result.first.size() == result.second.size());

  auto  matcherOn = Catch::Matchers::Approx(expectedOnsets);
  index margin = 1;
  matcherOn.margin(margin);

  CHECK_THAT(result.first, matcherOn);

  auto matcherOff = Catch::Matchers::Approx(expectedOffsets);
  matcherOff.margin(margin);

  CHECK_THAT(result.second, matcherOff);
}


TEST_CASE(
    "EnvelopeGate is predictable with smooth sine bursts and min time above",
    "[AmpGate][slicers]")
{
  auto sig = testsignals::smoothSine();
  auto expectedOnsets = std::vector<index>{2687, 24737};

  auto expectedOffsets = std::vector<index>{19429, 41479};

  auto params =
      TestParams{RampUp(5),         RampDown(25),        OnThreshold(-12),
                 OffThreshold(-12), MinLengthAbove(441), MinLengthBelow(1),
                 MinSliceLength(1), MinSilenceLength(1), LookAhead(0),
                 LookBack(0),       HighPassFreq(85)};


  auto result = runTest(sig, params);

  REQUIRE(result.first.size() == result.second.size());

  auto  matcherOn = Catch::Matchers::Approx(expectedOnsets);
  index margin = 1;
  matcherOn.margin(margin);

  CHECK_THAT(result.first, matcherOn);

  auto matcherOff = Catch::Matchers::Approx(expectedOffsets);
  matcherOff.margin(margin);

  CHECK_THAT(result.second, matcherOff);
}

TEST_CASE(
    "EnvelopeGate is predictable with smooth sine bursts and min time below",
    "[AmpGate][slicers]")
{
  auto sig = testsignals::smoothSine();
  auto expectedOnsets = std::vector<index>{1875, 23925};

  auto expectedOffsets = std::vector<index>{20219, 42269};

  auto params =
      TestParams{RampUp(5),         RampDown(25),        OnThreshold(-12),
                 OffThreshold(-12), MinLengthAbove(1),   MinLengthBelow(441),
                 MinSliceLength(1), MinSilenceLength(1), LookAhead(0),
                 LookBack(0),       HighPassFreq(85)};


  auto result = runTest(sig, params);

  REQUIRE(result.first.size() == result.second.size());

  auto  matcherOn = Catch::Matchers::Approx(expectedOnsets);
  index margin = 1;
  matcherOn.margin(margin);

  CHECK_THAT(result.first, matcherOn);

  auto matcherOff = Catch::Matchers::Approx(expectedOffsets);
  matcherOff.margin(margin);

  CHECK_THAT(result.second, matcherOff);
}

TEST_CASE("EnvelopeGate is predictable with smooth sine bursts and lookahead",
          "[AmpGate][slicers]")
{
  auto sig = testsignals::smoothSine();
  auto expectedOnsets = std::vector<index>{1875, 23925};

  auto expectedOffsets = std::vector<index>{20658, 42708};

  auto params =
      TestParams{RampUp(5),         RampDown(25),        OnThreshold(-12),
                 OffThreshold(-12), MinLengthAbove(1),   MinLengthBelow(1),
                 MinSliceLength(1), MinSilenceLength(1), LookAhead(441),
                 LookBack(0),       HighPassFreq(85)};


  auto result = runTest(sig, params);

  REQUIRE(result.first.size() == result.second.size());

  auto  matcherOn = Catch::Matchers::Approx(expectedOnsets);
  index margin = 1;
  matcherOn.margin(margin);

  CHECK_THAT(result.first, matcherOn);

  auto matcherOff = Catch::Matchers::Approx(expectedOffsets);
  matcherOff.margin(margin);

  CHECK_THAT(result.second, matcherOff);
}


TEST_CASE("EnvelopeGate is predictable with smooth sine bursts and lookback",
          "[AmpGate][slicers]")
{
  auto sig = testsignals::smoothSine();
  auto expectedOnsets = std::vector<index>{
      1435,  19499, 19568, 19638, 19707, 19776, 19846, 19915,
      19985, 20055, 20125, 20195, 23485, 41549, 41618, 41688,
      41757, 41826, 41896, 41965, 42035, 42105, 42175, 42245};

  auto expectedOffsets = std::vector<index>{
      19496, 19563, 19630, 19697, 19763, 19830, 19896, 19962,
      20027, 20091, 20155, 20218, 41546, 41613, 41680, 41747,
      41813, 41880, 41946, 42012, 42077, 42141, 42205, 42268};

  auto params =
      TestParams{RampUp(5),         RampDown(25),        OnThreshold(-12),
                 OffThreshold(-12), MinLengthAbove(1),   MinLengthBelow(1),
                 MinSliceLength(1), MinSilenceLength(1), LookAhead(0),
                 LookBack(441),     HighPassFreq(85)};


  auto result = runTest(sig, params);

  REQUIRE(result.first.size() == result.second.size());

  auto  matcherOn = Catch::Matchers::Approx(expectedOnsets);
  index margin = 1;
  matcherOn.margin(margin);

  CHECK_THAT(result.first, matcherOn);

  auto matcherOff = Catch::Matchers::Approx(expectedOffsets);
  matcherOff.margin(margin);

  CHECK_THAT(result.second, matcherOff);
}

TEST_CASE("EnvelopeGate is predictable with smooth sine bursts and both "
          "lookahead and lookback",
          "[AmpGate][slicers]")
{
  auto sig = testsignals::smoothSine();
  auto expectedOnsets = std::vector<index>{1654, 23704};

  auto expectedOffsets = std::vector<index>{20658, 42708};

  auto params =
      TestParams{RampUp(5),         RampDown(25),        OnThreshold(-12),
                 OffThreshold(-12), MinLengthAbove(1),   MinLengthBelow(1),
                 MinSliceLength(1), MinSilenceLength(1), LookAhead(441),
                 LookBack(221),     HighPassFreq(85)};


  auto result = runTest(sig, params);

  REQUIRE(result.first.size() == result.second.size());

  auto  matcherOn = Catch::Matchers::Approx(expectedOnsets);
  index margin = 1;
  matcherOn.margin(margin);

  CHECK_THAT(result.first, matcherOn);

  auto matcherOff = Catch::Matchers::Approx(expectedOffsets);
  matcherOff.margin(margin);

  CHECK_THAT(result.second, matcherOff);
}

TEST_CASE("EnvelopeGate is predictable on a real signal", "[AmpGate][slicers]")
{
  auto sig = testsignals::monoDrums();
  auto expectedOnsets = std::vector<index>{
      1269,   38394,  69830,  88034,  114533, 151761, 176327, 202300, 220866,
      239779, 252598, 276315, 283982, 326755, 353444, 372504, 390197, 417174};

  auto expectedOffsets = std::vector<index>{
      23328,  65751,  81905,  96307,  124975, 172742, 184798, 216023, 232738,
      249671, 272516, 283322, 310995, 343433, 366663, 384191, 398299, 428473};

  auto params =
      TestParams{RampUp(110),       RampDown(2205),         OnThreshold(-27),
                 OffThreshold(-31), MinLengthAbove(1),      MinLengthBelow(1),
                 MinSliceLength(1), MinSilenceLength(1100), LookAhead(0),
                 LookBack(441),     HighPassFreq(40)};


  auto result = runTest(sig, params);

  REQUIRE(result.first.size() == result.second.size());

  auto  matcherOn = Catch::Matchers::Approx(expectedOnsets);
  index margin = 1;
  matcherOn.margin(margin);

  CHECK_THAT(result.first, matcherOn);

  auto matcherOff = Catch::Matchers::Approx(expectedOffsets);
  matcherOff.margin(margin);

  CHECK_THAT(result.second, matcherOff);
}


} // namespace fluid
