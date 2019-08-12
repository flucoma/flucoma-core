#pragma once

#include "BufferedProcess.hpp"
#include "../common/AudioClient.hpp"
#include "../common/FluidBaseClient.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterTypes.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/ParameterTrackChanges.hpp"
#include "../nrt/FluidNRTClientWrapper.hpp"
#include "../../algorithms/public/SineExtraction.hpp"

#include <tuple>

namespace fluid {
namespace client {

class SinesClient : public FluidBaseClient, public AudioIn, public AudioOut
{
  enum SinesParamIndex {
    kBandwidth,
    kThreshold,
    kMinTrackLen,
    kMagWeight,
    kFreqWeight,
    kFFT,
    kMaxFFTSize
  };
public:

  FLUID_DECLARE_PARAMS(
    LongParam("bandwidth", "Bandwidth", 76, Min(1), FrameSizeUpperLimit<kFFT>()),
    FloatParam("threshold", "Threshold", 0.7, Min(0.0), Max(1.0)),
    LongParam("minTrackLen", "Min Track Length", 15, Min(0)),
    FloatParam("magWeight", "Magnitude Weighting", 0.01, Min(0.0), Max(1.0)),
    FloatParam("freqWeight", "Frequency Weighting", 0.5, Min(0.0), Max(1.0)),
    FFTParam<kMaxFFTSize>("fftSettings", "FFT Settings", 1024,-1,-1, FrameSizeLowerLimit<kBandwidth>()),
    LongParam<Fixed<true>>("maxFFTSize", "Maxiumm FFT Size", 16384, Min(4), PowerOfTwo{})
  );

  SinesClient(ParamSetViewType& p)
  : mParams(p), mSTFTBufferedProcess{static_cast<size_t>(get<kMaxFFTSize>()),1,2}
  {
    audioChannelsIn(1);
    audioChannelsOut(2);
  }

  template <typename T>
  void process(std::vector<HostVector<T>> &input, std::vector<HostVector<T>> &output, FluidContext& c,
               bool reset = false) {
    if (!input[0].data()) return;
    if (!output[0].data() && !output[1].data()) return;

    if (mTrackValues.changed(get<kFFT>().winSize(), get<kFFT>().hopSize(), get<kFFT>().fftSize(), get<kBandwidth>(), get<kMinTrackLen>()))
    {
      mSinesExtractor.reset(new algorithm::SineExtraction(get<kFFT>().winSize(), get<kFFT>().fftSize(), get<kFFT>().hopSize(),
                                                            get<kBandwidth>(), get<kThreshold>(), get<kMinTrackLen>(),
                                                            get<kMagWeight>(), get<kFreqWeight>()));
    } else
    {
      mSinesExtractor->setThreshold(get<kThreshold>());
      mSinesExtractor->setMagWeight(get<kMagWeight>());
      mSinesExtractor->setFreqWeight(get<kFreqWeight>());
      mSinesExtractor->setMinTrackLength(get<kMinTrackLen>());
    }

    mSTFTBufferedProcess.process(mParams, input, output, c, reset, [this](ComplexMatrixView in, ComplexMatrixView out) {
      mSinesExtractor->processFrame(in.row(0), out.transpose());
    });
  }

  size_t latency() { return get<kFFT>().winSize() + (get<kFFT>().hopSize() * get<kMinTrackLen>()); }

private:
  STFTBufferedProcess<ParamSetViewType, kFFT>  mSTFTBufferedProcess;
  std::unique_ptr<algorithm::SineExtraction> mSinesExtractor;
  ParameterTrackChanges<size_t,size_t,size_t,size_t,size_t> mTrackValues;
  size_t mWinSize{0};
  size_t mHopSize{0};
  size_t mFFTSize{0};
  size_t mBandwidth{0};
  size_t mMinTrackLen{0};
};

using RTSinesClient = ClientWrapper<SinesClient>;

auto constexpr NRTSineParams = makeNRTParams<RTSinesClient>({InputBufferParam("source", "Source Buffer")}, {BufferParam("sines","Sines Buffer"), BufferParam("residual", "Residual Buffer")});

using NRTSines = NRTStreamAdaptor<RTSinesClient, decltype(NRTSineParams), NRTSineParams, 1, 2>;

using NRTThreadedSines = NRTThreadingAdaptor<NRTSines>;

} // namespace client
} // namespace fluid
