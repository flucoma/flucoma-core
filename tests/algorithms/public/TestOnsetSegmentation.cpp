#define CATCH_CONFIG_MAIN

#include "SlicerTestHarness.hpp"
#include <TestUtils.hpp> 
#include <algorithms/public/OnsetSegmentation.hpp>
#include <algorithms/public/STFT.hpp>
#include <catch2/catch.hpp>
#include <data/FluidIndex.hpp>
#include <data/FluidTensor.hpp>
#include <Signals.hpp>
#include <algorithm>
#include <array>
#include <complex>
#include <functional>
#include <string>
#include <vector>


namespace fluid {

using testsignals::drums;
using testsignals::monoDrums;
using testsignals::monoImpulses;
using testsignals::oneImpulse;
using testsignals::stereoImpulses;

std::vector<index> spikeExpected{22050};

using Window = StrongType<index, struct WindowTag>;
using Hop = StrongType<index, struct HopTag>;
using FFT = StrongType<index, struct FFTTag>;
using Metric = StrongType<index, struct MetricTag>;
using MinSliceLen = StrongType<index, struct MinSliceLenTag>;
using FilterSize = StrongType<index, struct FilterSizeTag>;
using Threshold = StrongType<double, struct ThresholdTag>;
using FrameDelta = StrongType<index, struct FrameDeltaTag>;
using Data = StrongType<FluidTensorView<const double, 1>, struct DataTag>;
using Expected = StrongType<std::vector<index>, struct ExpectedTag>;
using Margin = StrongType<index, struct MarginTag>;

struct TestParams
{
  Window      window;
  Hop         hop;
  FFT         fft;
  Metric      metric;
  MinSliceLen minSlice;
  FilterSize  filterSize;
  Threshold   threshold;
  FrameDelta  frameDelta{1};
  Data        data;
  Expected    expected;
  Margin      margin;
};


std::vector<index> runOneTest(const TestParams& params)
{
  auto makeSlicer = [](const TestParams& p) {
    auto res = algorithm::OnsetSegmentation(p.fft);
    res.init(p.window, p.fft, p.filterSize);
    return res;
  };

  auto invokeSlicer = [](algorithm::OnsetSegmentation& slicer, auto source,
                         const TestParams& p) {
    return slicer.processFrame(source, p.metric, p.filterSize, p.threshold,
                               p.minSlice, p.frameDelta);
  };

  auto inputFn = [](auto source) { return source; };
  return SlicerTestHarness(params.data, params, makeSlicer, inputFn,
                           invokeSlicer, 0); // params.hop - (params.window/2));
}


TEST_CASE("OnsetSegmentation can produce the same results as SC tests",
          "[OnsetSegmentation][slicers]")
{

  SECTION("single impulse tests")
  {


    auto metric = GENERATE(as<Metric>{}, 0, 1, 2, 3, 4, 8, 9);
    auto params =
        TestParams{Window(1024),
                   Hop(512),
                   FFT(1024),
                   metric,
                   MinSliceLen(2),
                   FilterSize(5),
                   Threshold(0.5),
                   FrameDelta(0),
                   Data(FluidTensorView<const double, 1>(oneImpulse())),
                   Expected(spikeExpected),
                   Margin(34)};

    INFO("Click Test with Metric " << index(params.metric));

    auto                      result = runOneTest(params);
    const std::vector<index>& points = params.expected();
    auto                      matcher = Catch::Matchers::Approx(points);
    index                     margin = params.margin;
    matcher.margin(margin);

    CHECK(result.size() == points.size());
    CHECK_THAT(result, matcher);
  }

  SECTION("stereo impulses")
  {

    auto params = TestParams{Window(512),
                             Hop(64),
                             FFT(512),
                             Metric(9),
                             MinSliceLen(2),
                             FilterSize(5),
                             Threshold(0.1),
                             FrameDelta(0),
                             Data(monoImpulses()),
                             Expected({1000, 12025, 23051, 34076}),
                             Margin(64)};

    auto                      result = runOneTest(params);
    const std::vector<index>& points = params.expected();
    auto                      matcher = Catch::Matchers::Approx(points);
    index                     margin = params.margin;
    matcher.margin(margin);

    INFO("stereo impulse test")
    CHECK(result.size() == points.size());
    CHECK_THAT(result, matcher);
  }


  SECTION("drum tests")
  {

    struct LabelledParams
    {
      std::string label;
      TestParams  p;
    };

    auto makeDrumParams = [](std::string label, Metric m, Threshold t, Window w,
                             Hop h, FFT f, MinSliceLen l, Expected e) {
      return LabelledParams{
          label, TestParams{w, h, f, m, l, FilterSize(5), t, FrameDelta(0),
                            Data(FluidTensorView<const double, 1>(monoDrums())),
                            e, Margin(1)}};
    };

    auto params = GENERATE_REF(
        makeDrumParams(
            "test_drums_energy", Metric(0), Threshold(0.5), Window(1024),
            Hop(512), FFT(1024), MinSliceLen(2),
            Expected({1536,   8192,   38400,  51200,  69632,  88064,  114688,
                      151552, 157184, 176640, 202240, 220672, 240128, 252416,
                      259072, 276480, 283648, 289792, 296448, 302592, 327168,
                      353280, 372736, 390144, 417280})),
        makeDrumParams("test_drums_hfc", Metric(1), Threshold(20), Window(512),
                       Hop(128), FFT(512), MinSliceLen(20),
                       Expected({1792,   7936,   38784,  51200,  70016,  88320,
                                 114688, 151808, 157568, 176768, 202368, 221056,
                                 240256, 252800, 259328, 284032, 289792, 302976,
                                 327040, 353536, 372608, 390528, 417280})),
        makeDrumParams(
            "test_drums_SpectralFlux", Metric(2), Threshold(0.2), Window(1000),
            Hop(220), FFT(1024), MinSliceLen(2),
            Expected({1760,   8580,   38720,  51260,  69960,  88220,
                      114620, 151800, 157520, 176660, 202400, 221100,
                      240240, 252780, 259380, 284020, 289740, 296560,
                      302940, 326920, 353540, 372680, 390500, 417340})),
        makeDrumParams(
            "test_drums_MKL", Metric(3), Threshold(2), Window(800), Hop(330),
            FFT(1024), MinSliceLen(2),
            Expected({0,      1650,   69960,  88110,  100650, 114510, 127050,
                      146190, 151800, 189420, 202290, 220770, 239910, 252780,
                      259380, 277530, 283800, 289740, 302940, 326700, 353430,
                      372570, 378840, 403920, 417120, 429990, 449130})),
        makeDrumParams(
            "test_drums_cosine", Metric(5), Threshold(0.2), Window(1000),
            Hop(200), FFT(1024), MinSliceLen(5),
            Expected({0,      1600,   38400,  51200,  69800,  88000,
                      146200, 151800, 157400, 176400, 202200, 240000,
                      243000, 252600, 276400, 280600, 289800, 302800,
                      326800, 353400, 390200, 417200, 449000, 453600})),
        makeDrumParams(
            "test_drums_phase_dev", Metric(6), Threshold(0.1), Window(2000),
            Hop(200), FFT(2048), MinSliceLen(5),
            Expected({2200, 8800, 40200, 51600, 70600, 115200, 152200, 158400,
                      202800, 241000, 253400, 259800, 290200, 303400, 327600,
                      354000, 373200, 417600})),
        makeDrumParams(
            "test_drums_Wphase_dev", Metric(7), Threshold(0.1), Window(1500),
            Hop(300), FFT(2048), MinSliceLen(5),
            Expected({1800,   8400,   38700,  51300,  70200,  88200,  114600,
                      151800, 157500, 165000, 176700, 202500, 221100, 240300,
                      252900, 259500, 276900, 284100, 289800, 296400, 303000,
                      327300, 353700, 372600, 390600, 417300})),
        makeDrumParams("test_drums_complex", Metric(8), Threshold(0.1),
                       Window(512), Hop(50), FFT(512), MinSliceLen(50),
                       Expected({1750,   5200,   7900,   38550,  51200,  54650,
                                 69900,  88150,  114600, 151800, 154550, 157300,
                                 176550, 202350, 206100, 220950, 240050, 252700,
                                 259350, 276800, 283900, 289750, 296350, 302850,
                                 326900, 353500, 372600, 390350, 417250})),
        makeDrumParams("test_drums_Rcomplex", Metric(9), Threshold(0.2),
                       Window(1950), Hop(40), FFT(2048), MinSliceLen(50),
                       Expected({2040,   9000,   39560,  51760,  89200,  115000,
                                 152200, 158280, 177480, 202720, 240760, 253120,
                                 259800, 277840, 284880, 290200, 297560, 303360,
                                 327440, 354040, 373560, 391360, 417600})));

    INFO("" << params.label);

    auto result = runOneTest(params.p);

    CHECK(result.size() == params.p.expected().size());
    CHECK_THAT(result, Catch::Equals(params.p.expected()));
  }

  SECTION("Test Filtersize")
  {

    auto makeDrumParams = [](FilterSize f, Expected e) {
      return TestParams{Window(512),
                        Hop(50),
                        FFT(512),
                        Metric(8),
                        MinSliceLen(50),
                        f,
                        Threshold(0.1),
                        FrameDelta(0),
                        Data(FluidTensorView<const double, 1>(monoDrums())),
                        e,
                        Margin(1)};
    };

    auto params = GENERATE_REF(
        makeDrumParams(
            FilterSize(3),
            Expected({1750,   5200,   8400,   38800,  51200,  53750,  69950,
                      88200,  114600, 151800, 157300, 176750, 202350, 204950,
                      220950, 240150, 252700, 259350, 276800, 284000, 289750,
                      296550, 302850, 326950, 353500, 372600, 390450, 417250})),
        makeDrumParams(
            FilterSize(7),
            Expected({1750,   5200,   7850,   38550,  51200,  54650,
                      69900,  76800,  88150,  100800, 114600, 151800,
                      157300, 166200, 176550, 202350, 206100, 220950,
                      228450, 240050, 252700, 259350, 276800, 283900,
                      289750, 296350, 302850, 326900, 332050, 353500,
                      372600, 379000, 390350, 404050, 417250, 430250})),
        makeDrumParams(
            FilterSize(29),
            Expected({1750,   7850,   13750,  38550,  46400,  51200,  69900,
                      76800,  88150,  100800, 114600, 127300, 151800, 157300,
                      164350, 176550, 189650, 202350, 220950, 228400, 240000,
                      252700, 259350, 276600, 283900, 289750, 296350, 302850,
                      326900, 332050, 353500, 372600, 379000, 390350, 404050,
                      417250, 430200, 449350})));

    INFO("Filter Size " << index(params.filterSize));
    auto result = runOneTest(params);
    CHECK(result.size() == params.expected().size());
    CHECK_THAT(result, Catch::Equals(params.expected()));
  }

  SECTION("frame delta")
  {
    auto params = TestParams{
        Window(1000),
        Hop(200),
        FFT(1024),
        Metric(5),
        MinSliceLen(5),
        FilterSize(7),
        Threshold(0.2),
        FrameDelta(100),
        Data(monoDrums()),
        Expected({0,      1600,   38400,  51200,  69800,  88000,  114600,
                  146200, 151800, 157400, 176400, 202200, 240000, 243000,
                  252600, 276400, 278400, 280600, 289800, 302800, 326800,
                  353400, 390200, 404000, 417200, 449000, 453600}),
        Margin(1)};

    INFO("frame delta test");
    auto result = runOneTest(params);
    CHECK(result.size() == params.expected().size());
    CHECK_THAT(result, Catch::Equals(params.expected()));
  }
}

} // namespace fluid
