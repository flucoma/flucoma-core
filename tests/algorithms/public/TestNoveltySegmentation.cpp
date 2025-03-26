#define CATCH_CONFIG_MAIN
#include <flucoma/algorithms/public/DCT.hpp>
#include <flucoma/algorithms/public/Loudness.hpp>
#include <flucoma/algorithms/public/MelBands.hpp>
#include <flucoma/algorithms/public/NoveltySegmentation.hpp>
#include <flucoma/algorithms/public/STFT.hpp>
#include <flucoma/algorithms/public/YINFFT.hpp>
#include <catch2/catch.hpp>
#include <flucoma/data/FluidIndex.hpp>
#include <flucoma/data/FluidTensor.hpp>
#include <Signals.hpp>
#include <algorithm>
#include <array>
#include <complex>
#include <string>
#include <vector>


using fluid::FluidTensor;
using fluid::FluidTensorView;
using fluid::Slice;
using fluid::algorithm::NoveltySegmentation;
using fluid::algorithm::STFT;


static std::string audio_path;

struct Params
{
  fluid::index window;
  fluid::index hop;
  fluid::index fft;
  fluid::index minSlice;
  fluid::index kernel;
  fluid::index filter;
  fluid::index dims;
  double       threshold;
};


namespace fluid {

template <typename F>
std::vector<index> NoveltyTestHarness(FluidTensorView<const double, 1> testSignal,
                                      Params p, F&& f)
{

  // NB: This has a lot of arcana about padding amounts and adjustments to get
  // the same results as the equivalent tests already implemented in SC.
  // However, I think the latency calculation in NoveltySlice, and the padding
  // assumptions in NRTWrapper could do with another look, as I think we can get
  // closer than we do

  const index filt = p.filter % 2 ? p.filter + 1 : p.filter;
  const index halfWindow = p.window; // >> 1;
  const index padding = p.hop * (((p.kernel + 1) >> 1) + (filt >> 1));
  FluidTensor<double, 1> padded(p.window + halfWindow + padding +
                                testSignal.size());
  padded.fill(0);
  padded(Slice(halfWindow, testSignal.size())) <<= testSignal;
  const fluid::index nHops =
      std::floor<index>((padded.size() - p.window) / p.hop);

  auto slicer = NoveltySegmentation(p.kernel, p.fft / 2 + 1, p.filter);
  slicer.init(p.kernel, p.filter, p.dims); // sigh

  std::vector<index> spikePositions;

  for (index i = 0; i < nHops; ++i)
  {
    auto input = f(padded(Slice(i * p.hop, p.window)));
    if (slicer.processFrame(input, p.threshold, p.minSlice) > 0)
    {
      spikePositions.push_back((i * p.hop) - padding - p.hop);
    }
  }

  // This reproduces what the NRT wrapper does (and hence the result that the
  // existing test in SC sees). I'm dubious that
  //  it really ought to be needed though. I think we're adjusting the latency
  //  by a hop too much
  std::transform(spikePositions.begin(), spikePositions.end(),
                 spikePositions.begin(),
                 [&p](index x) { return std::max<index>(0, x); });

  spikePositions.erase(
      std::unique(spikePositions.begin(), spikePositions.end()),
      spikePositions.end());

  return spikePositions;
}
} // namespace fluid

std::vector<fluid::index>
NoveltySTFTTest(fluid::FluidTensorView<const double, 1> testSignal, Params p)
{
  FluidTensor<std::complex<double>, 1> stftFrame(p.dims);
  FluidTensor<double, 1>               magnitudes(p.dims);

  auto stft = STFT{p.window, p.fft, p.hop};

  auto makeInput = [&stft, &stftFrame, &magnitudes](auto source) {
    stft.processFrame(source, stftFrame);
    stft.magnitude(stftFrame, magnitudes);
    return fluid::FluidTensorView<double, 1>(magnitudes);
  };

  return NoveltyTestHarness(testSignal, p, makeInput);
}


std::vector<fluid::index>
NoveltyMFCCTest(fluid::FluidTensorView<const double, 1> testSignal, Params p)
{
  FluidTensor<std::complex<double>, 1> stftFrame((p.fft / 2) + 1);
  FluidTensor<double, 1>               magnitudes((p.fft / 2) + 1);
  FluidTensor<double, 1>               melFrame(40);
  FluidTensor<double, 1>               mfccFrame(13);

  auto stft = STFT{p.window, p.fft, p.hop};
  auto mels = fluid::algorithm::MelBands(40, p.fft);
  auto dct = fluid::algorithm::DCT(40, 13);

  // The NoveltySliceClient inits mels only up to 2k, which I'm not is correct
  mels.init(20, 2000, 40, (p.fft / 2) + 1, 44100, p.window);
  dct.init(40, 13);
  auto makeInput = [&stft, &mels, &dct, &stftFrame, &magnitudes, &melFrame,
                    &mfccFrame](auto source) {
    stft.processFrame(source, stftFrame);
    stft.magnitude(stftFrame, magnitudes);
    mels.processFrame(magnitudes, melFrame, false, false, true, fluid::FluidDefaultAllocator());
    dct.processFrame(melFrame, mfccFrame);
    return fluid::FluidTensorView<double, 1>(mfccFrame);
  };

  return NoveltyTestHarness(testSignal, p, makeInput);
}

std::vector<fluid::index>
NoveltyPitchTest(fluid::FluidTensorView<const double, 1> testSignal, Params p)
{
  FluidTensor<std::complex<double>, 1> stftFrame((p.fft / 2) + 1);
  FluidTensor<double, 1>               magnitudes((p.fft / 2) + 1);
  FluidTensor<double, 1>               pitchFrame(2);

  auto stft = STFT{p.window, p.fft, p.hop};
  auto pitch = fluid::algorithm::YINFFT((p.fft / 2) + 1);

  auto makeInput = [&stft, &pitch, &stftFrame, &magnitudes,
                    &pitchFrame](auto source) {
    stft.processFrame(source, stftFrame);
    stft.magnitude(stftFrame, magnitudes);
    pitch.processFrame(magnitudes, pitchFrame, 20, 5000, 44100);
    return fluid::FluidTensorView<double, 1>(pitchFrame);
  };

  return NoveltyTestHarness(testSignal, p, makeInput);
}

std::vector<fluid::index>
NoveltyLoudnessTest(fluid::FluidTensorView<const double, 1> testSignal, Params p)
{
  FluidTensor<double, 1> loudnessFrame(2);

  auto loudness = fluid::algorithm::Loudness{p.fft};
  loudness.init(p.window, 44100);

  auto makeInput = [&loudness, &loudnessFrame](auto source) {
    loudness.processFrame(source, loudnessFrame, true, true);
    return fluid::FluidTensorView<double, 1>(loudnessFrame);
  };

  return NoveltyTestHarness(testSignal, p, makeInput);
}

TEST_CASE("NoveltySegmentation will segment on clicks with some predictability",
          "[Novelty][slicers]")
{

  using fluid::index;

  auto monoInput = fluid::testsignals::monoImpulses();

  Params p;
  p.window = 128;
  p.fft = 128;
  p.hop = 64;
  p.threshold = 0.5;
  p.minSlice = 2;
  p.kernel = 3;
  p.filter = 1;
  p.dims = (p.fft / 2) + 1;

  // FluidTensor<double, 1> monoInput(testSignal.cols());
  // monoInput = testSignal.row(0);
  // monoInput.apply(testSignal.row(1), [](double& x, double y) { x += y; });

  const std::vector<index> spikePositions = NoveltySTFTTest(monoInput, p);

  const std::vector<index> expected{1000, 12025, 23051, 34076};

  auto  matcher = Catch::Matchers::Approx(expected);
  index margin = 128;
  matcher.margin(margin);

  REQUIRE(spikePositions.size() == 4);
  REQUIRE_THAT(spikePositions, matcher);
}

TEST_CASE("NoveltySegmentation will segment sine bursts STFT mags accurately",
          "[Novelty][slicers]")
{
  using fluid::index;
  Params p;
  p.window = 512;
  p.fft = 1024;
  p.hop = 256;
  p.threshold = 0.38;
  p.minSlice = 4;
  p.kernel = 3;
  p.filter = 1;
  p.dims = (p.fft / 2) + 1;

  const auto testSignal = fluid::testsignals::sharpSines();

  const std::vector<index> spikePositions = NoveltySTFTTest(testSignal, p);

  const std::vector<index> expected{512, 11008, 22016, 33024};
  REQUIRE(spikePositions.size() == 4);
  REQUIRE_THAT(spikePositions, Catch::Matchers::Equals(expected));
}

TEST_CASE(
    "NoveltySegmentation will do something predictable with a smooth AM sine",
    "[Novelty][slicers]")
{
  using fluid::index;
  Params p;
  p.window = 512;
  p.fft = 1024;
  p.hop = 256;
  p.threshold = 0.34;
  p.minSlice = 30;
  p.kernel = 3;
  p.filter = 1;
  p.dims = (p.fft / 2) + 1;

  const auto testSignal = fluid::testsignals::smoothSine();

  const std::vector<index> spikePositions = NoveltySTFTTest(testSignal, p);

  const std::vector<index> expected{0, 22016};
  REQUIRE(spikePositions.size() == 2);
  REQUIRE_THAT(spikePositions, Catch::Matchers::Equals(expected));
}

TEST_CASE("NoveltySegmentation behaves with different filter sizes","[Novelty][slicers]"){

  using fluid::index;

  struct Settings
  {
    index              filterSize;
    std::vector<index> expected;
  };

  auto settings = GENERATE(
      Settings{1,
               {0, 292352, 558592, 563712, 617984, 669696, 722432, 774656,
                826368, 973824, 1000960}},
      Settings{4,
               {0, 292352, 564224, 617984, 670208, 722944, 774656, 826880,
                974848, 1000960}},
      Settings{12,
               {512, 292352, 564224, 617984, 723456, 774656, 827392, 1000960}});

  Params p;
  p.window = 1024;
  p.fft = 1024;
  p.hop = 512;
  p.threshold = 0.1;
  p.minSlice = 2;
  p.kernel = 31;
  p.filter = settings.filterSize;
  p.dims = (p.fft / 2) + 1;

  const auto testSignal = fluid::testsignals::guitarStrums();

  const std::vector<index> spikePositions = NoveltySTFTTest(testSignal.row(0), p);
  CHECK(spikePositions.size() == settings.expected.size());
  REQUIRE_THAT(spikePositions, Catch::Matchers::Equals(settings.expected));
}

TEST_CASE("NoveltySegmentation works with MFCC feature","[Novelty][slicers]"){

  using fluid::index;

  std::vector<index> expected{320,    34880,  105856, 117504, 179200,
                              186496, 205248, 223936, 238208, 256448,
                              346944, 352512, 368512, 401088, 414016,
                              455424, 465600, 481728, 494784, 512640};

  Params p;
  p.window = 2048;
  p.fft = 2048;
  p.hop = 64;
  p.threshold = 0.6;
  p.minSlice = 50;
  p.kernel = 17;
  p.filter = 5;
  p.dims = 13;

  const auto testSignal = fluid::testsignals::eurorackSynth();

  const std::vector<index> spikePositions = NoveltyMFCCTest(testSignal.row(0), p);
  CHECK(spikePositions.size() == expected.size());
  REQUIRE_THAT(spikePositions, Catch::Matchers::Equals(expected));
}

TEST_CASE("NoveltySegmentation works with pitch feature","[Novelty][slicers]"){
  

  using fluid::index;

  std::vector<index> expected{
      128,    34880,  47360,  145280, 181888, 186496, 191040, 195648, 200320,
      204928, 230976, 266880, 349056, 354944, 358784, 362688, 367552, 371456,
      375360, 414080, 425728, 465600, 471616, 481664, 487744, 492992};

  Params p;
  p.window = 2048;
  p.fft = 2048;
  p.hop = 64;
  p.threshold = 0.2;
  p.minSlice = 50;
  p.kernel = 9;
  p.filter = 5;
  p.dims = 2;

  const auto testSignal = fluid::testsignals::eurorackSynth();

  const std::vector<index> spikePositions = NoveltyPitchTest(testSignal.row(0), p);
  CHECK(spikePositions.size() == expected.size());
  REQUIRE_THAT(spikePositions, Catch::Matchers::Equals(expected));
}

// TEST_CASE("NoveltySegmentation works with loudness feature","[Novelty][slicers]"){
  
//   using fluid::index;

//   std::vector<index> expected{0,      19008,  24640,  34624,  58240,  117696,
//                               122048, 179392, 229376, 256832, 260288, 265536,
//                               287488, 306752, 335616, 401280, 413888, 464896,
//                               471936, 477184, 483456, 488064, 493376, 513664};

//   Params p;
//   p.window = 2048;
//   p.fft = 2048;
//   p.hop = 64;
//   p.threshold = 0.0145;
//   p.minSlice = 50;
//   p.kernel = 17;
//   p.filter = 5;
//   p.dims = 2;

//   const auto testSignal = fluid::testsignals::eurorackSynth();

//   const std::vector<index> spikePositions = NoveltyLoudnessTest(testSignal.row(0), p);
//   CHECK(spikePositions.size() == expected.size());
//   REQUIRE_THAT(spikePositions, Catch::Matchers::Equals(expected));
// }
