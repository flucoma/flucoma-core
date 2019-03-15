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
    LongParam("bw", "Bandwidth", 76, Min(1)),
    FloatParam("thresh", "Threshold", 0.7, Min(0.0), Max(1.0)),
    LongParam("minTrackLen", "Min Track Length", 15, Min(0)),
    FloatParam("magWeight", "Magnitude Weighting", 0.1, Min(0.0), Max(1.0)),
    FloatParam("freqWeight", "Frequency Weighting", 0.1, Min(0.0), Max(1.0)),
    FFTParam<kMaxFFTSize>("fft", "FFT Settings", 1024,-1,-1),
    LongParam<Fixed<true>>("maxFFTSize", "Maxiumm FFT Size", 16384));

//using ParamsT = decltype(SinesParams);

template <typename T>
class SinesClient : public FluidBaseClient<decltype(SinesParams), SinesParams>, public AudioIn, public AudioOut
{
  using HostVector = HostVector<T>;

public:
  SinesClient(ParamSetType& p)
  : FluidBaseClient(p), mSTFTBufferedProcess{param<kMaxFFTSize>(p),1,2}
  {
    FluidBaseClient::audioChannelsIn(1);
    FluidBaseClient::audioChannelsOut(2);
  }

  void process(std::vector<HostVector> &input, std::vector<HostVector> &output)
  {

    if (!input[0].data()) return;
    if (!output[0].data() && !output[1].data()) return;

    if (mTrackValues.changed(param<kFFT>(mParams).winSize(), param<kFFT>(mParams).hopSize(), param<kFFT>(mParams).fftSize(), param<kBandwidth>(mParams), param<kMinTrackLen>(mParams)))
    {
      mSinesExtractor.reset(new algorithm::RTSineExtraction(param<kFFT>(mParams).winSize(), param<kFFT>(mParams).fftSize(), param<kFFT>(mParams).hopSize(),
                                                            param<kBandwidth>(mParams), param<kThreshold>(mParams), param<kMinTrackLen>(mParams),
                                                            param<kMagWeight>(mParams), param<kFreqWeight>(mParams)));
    } else
    {
      mSinesExtractor->setThreshold(param<kThreshold>(mParams));
      mSinesExtractor->setMagWeight(param<kMagWeight>(mParams));
      mSinesExtractor->setFreqWeight(param<kFreqWeight>(mParams));
      mSinesExtractor->setMinTrackLength(param<kMinTrackLen>(mParams));
    }

    mSTFTBufferedProcess.process(mParams, input, output, [this](ComplexMatrixView in, ComplexMatrixView out) {
      mSinesExtractor->processFrame(in.row(0), out.transpose());
    });
  }

  size_t latency() { return param<kFFT>(mParams).winSize() + (param<kFFT>(mParams).hopSize() * param<kMinTrackLen>(mParams)); }

private:
  STFTBufferedProcess<ParamSetType, T, kFFT>  mSTFTBufferedProcess;
  std::unique_ptr<algorithm::RTSineExtraction>   mSinesExtractor;
  ParameterTrackChanges<size_t,size_t,size_t,size_t,size_t> mTrackValues;
  size_t mWinSize{0};
  size_t mHopSize{0};
  size_t mFFTSize{0};
  size_t mBandwidth{0};
  size_t mMinTrackLen{0};
};

/*
template <typename Params, typename T, typename U>
using NRTSines = NRTStreamAdaptor<SinesClient,Params,T,U,1,2>;

auto constexpr NRTSineParams = impl::makeNRTParams({BufferParam("srcBuf", "Source Buffer")}, {BufferParam("sinesBuf","Sines Buffer"), BufferParam("resBuf", "Residual Buffer")}, SinesParams);
*/

} // namespace client
} // namespace fluid
