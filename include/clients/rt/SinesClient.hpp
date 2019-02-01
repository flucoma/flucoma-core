#pragma once

#include <algorithms/public/RTSineExtraction.hpp>
#include <clients/common/AudioClient.hpp>
#include <clients/common/FluidBaseClient.hpp>
#include <clients/common/ParameterConstraints.hpp>
#include <clients/common/ParameterTypes.hpp>
#include <clients/common/ParameterTrackChanges.hpp>
#include <clients/rt/BufferedProcess.hpp>
#include <tuple>

namespace fluid {
namespace client {

enum SinesParamIndex {
  kBandwidth,
  kThreshold,
  kMinTrackLen,
  kMagWeight,
  kFreqWeight,
  kWinSize,
  kHopSize,
  kFFTSize,
  kMaxWinSize
};

auto constexpr SinesParams = std::make_tuple(
    LongParam("bandwidth", "Bandwidth", 76, Min(1)), FloatParam("threshold", "Threshold", 0.7, Min(0.0), Max(1.0)),
    LongParam("minTrackLen", "Min Track Length", 15, Min(0)),
    FloatParam("magWeight", "Magnitude Weighting", 0.1, Min(0.0), Max(1.0)),
    FloatParam("freqWeight", "Frequency Weighting", 0.1, Min(0.0), Max(1.0)),
    LongParam("winSize", "Window Size", 1024, Min(4)), LongParam("hopSize", "Hop Size", 512),
    LongParam("fftSize", "FFT Size", 8192, LowerLimit<kWinSize>(), PowerOfTwo()),
    LongParam("maxWinSize", "Maxiumm Window Size", 16384));

using ParamsT = decltype(SinesParams);

template <typename T, typename U = T> class SinesClient : public FluidBaseClient<ParamsT>, public AudioIn, public AudioOut
{
  using HostVector = HostVector<U>;

public:
  SinesClient()
      : FluidBaseClient<ParamsT>(SinesParams)
  {
    audioChannelsIn(1);
    audioChannelsOut(2);
  }

  SinesClient(SinesClient &) = delete;
  SinesClient operator=(SinesClient &) = delete;

  // Here we do an STFT and its inverse
  void process(std::vector<HostVector> &input, std::vector<HostVector> &output)
  {

    if (!input[0].data()) return;
    if (!output[0].data() && !output[1].data()) return;

    if (mTrackValues.changed(get<kWinSize>(), get<kHopSize>(), get<kFFTSize>(), get<kBandwidth>(), get<kMinTrackLen>()))
    {
      mSinesExtractor.reset(new algorithm::RTSineExtraction(get<kWinSize>(), get<kFFTSize>(), get<kHopSize>(),
                                                            get<kBandwidth>(), get<kThreshold>(), get<kMinTrackLen>(),
                                                            get<kMagWeight>(), get<kFreqWeight>()));
    } else
    {
      mSinesExtractor->setThreshold(get<kThreshold>());
      mSinesExtractor->setMagWeight(get<kMagWeight>());
      mSinesExtractor->setFreqWeight(get<kFreqWeight>());
      mSinesExtractor->setMinTrackLength(get<kMinTrackLen>());
    }

    mSTFTBufferedProcess.process(*this, input, output, [this](ComplexMatrixView in, ComplexMatrixView out) {
      mSinesExtractor->processFrame(in.row(0), out.transpose());
    });
  }

  size_t latency() { return get<kHopSize>() * get<kMinTrackLen>(); }

private:
  STFTBufferedProcess<T, U, SinesClient, kMaxWinSize, kWinSize, kHopSize, kFFTSize, true> mSTFTBufferedProcess;
  std::unique_ptr<algorithm::RTSineExtraction>                                            mSinesExtractor;
  ParameterTrackChanges<size_t,size_t,size_t,size_t,size_t> mTrackValues;
  size_t mWinSize{0};
  size_t mHopSize{0};
  size_t mFFTSize{0};
  size_t mBandwidth{0};
  size_t mMinTrackLen{0};
};

} // namespace client
} // namespace fluid

