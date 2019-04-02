#pragma once

#include <algorithms/public/RTSineExtraction.hpp>
#include <clients/common/AudioClient.hpp>
#include <clients/common/FluidBaseClient.hpp>
#include <clients/nrt/FluidNRTClientWrapper.hpp>
#include <clients/common/ParameterConstraints.hpp>
#include <clients/common/ParameterTypes.hpp>
#include <clients/common/ParameterSet.hpp>
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
  kFFT,
  kMaxFFTSize
};

extern auto constexpr SinesParams = defineParameters(
    LongParam("bandwidth", "Bandwidth", 76, Min(1)),
    FloatParam("threshold", "Threshold", 0.7, Min(0.0), Max(1.0)),
    LongParam("minTrackLen", "Min Track Length", 15, Min(0)),
    FloatParam("magWeight", "Magnitude Weighting", 0.1, Min(0.0), Max(1.0)),
    FloatParam("freqWeight", "Frequency Weighting", 0.1, Min(0.0), Max(1.0)),
    FFTParam<kMaxFFTSize>("fftSettings", "FFT Settings", 1024,-1,-1),
    LongParam<Fixed<true>>("maxFFTSize", "Maxiumm FFT Size", 16384, Min(4), PowerOfTwo{})
  );


template <typename T>
class SinesClient : public FluidBaseClient<decltype(SinesParams), SinesParams>, public AudioIn, public AudioOut
{
  using HostVector = HostVector<T>;

public:
  SinesClient(ParamSetViewType& p)
  : FluidBaseClient(p), mSTFTBufferedProcess{get<kMaxFFTSize>(),1,2}
  {
    FluidBaseClient::audioChannelsIn(1);
    FluidBaseClient::audioChannelsOut(2);
  }

  void process(std::vector<HostVector> &input, std::vector<HostVector> &output)
  {

    if (!input[0].data()) return;
    if (!output[0].data() && !output[1].data()) return;

    if (mTrackValues.changed(get<kFFT>().winSize(), get<kFFT>().hopSize(), get<kFFT>().fftSize(), get<kBandwidth>(), get<kMinTrackLen>()))
    {
      mSinesExtractor.reset(new algorithm::RTSineExtraction(get<kFFT>().winSize(), get<kFFT>().fftSize(), get<kFFT>().hopSize(),
                                                            get<kBandwidth>(), get<kThreshold>(), get<kMinTrackLen>(),
                                                            get<kMagWeight>(), get<kFreqWeight>()));
    } else
    {
      mSinesExtractor->setThreshold(get<kThreshold>());
      mSinesExtractor->setMagWeight(get<kMagWeight>());
      mSinesExtractor->setFreqWeight(get<kFreqWeight>());
      mSinesExtractor->setMinTrackLength(get<kMinTrackLen>());
    }

    mSTFTBufferedProcess.process(mParams, input, output, [this](ComplexMatrixView in, ComplexMatrixView out) {
      mSinesExtractor->processFrame(in.row(0), out.transpose());
    });
  }

  size_t latency() { return get<kFFT>().winSize() + (get<kFFT>().hopSize() * get<kMinTrackLen>()); }

private:
  STFTBufferedProcess<ParamSetViewType, T, kFFT>  mSTFTBufferedProcess;
  std::unique_ptr<algorithm::RTSineExtraction>   mSinesExtractor;
  ParameterTrackChanges<size_t,size_t,size_t,size_t,size_t> mTrackValues;
  size_t mWinSize{0};
  size_t mHopSize{0};
  size_t mFFTSize{0};
  size_t mBandwidth{0};
  size_t mMinTrackLen{0};
};

auto constexpr NRTSineParams = makeNRTParams<SinesClient>({BufferParam("source", "Source Buffer")}, {BufferParam("sines","Sines Buffer"), BufferParam("residual", "Residual Buffer")});

template <typename T>
using NRTSines = NRTStreamAdaptor<SinesClient<T>, decltype(NRTSineParams), NRTSineParams, 1, 2>;


} // namespace client
} // namespace fluid
