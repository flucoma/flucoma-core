#pragma once


#include <clients/common/ParameterTypes.hpp>
#include <clients/common/ParameterConstraints.hpp>
#include <clients/common/FluidBaseClient.hpp>
#include <clients/common/SpikesToTimes.hpp>
#include <algorithms/public/NoveltySegmentation.hpp>
#include <algorithms/public/STFT.hpp>

namespace fluid {
namespace client {

enum NoveltyParamIndex {kSource, kOffset, kNumFrames, kStartChan, kNumChans, kTransBuf, kKernelSize, kThreshold,kFilterSize,kFFT};

auto constexpr NoveltyParams =defineParameters(
  BufferParam("srcBuf","Source Buffer"),
  LongParam("startAt","Source Offset",0,Min(0)),
  LongParam("nFrames","Number of Frames",-1),
  LongParam("startChan","Start Channel",0,Min(0)),
  LongParam("numChans","Number of Channels",-1),
  BufferParam("indBuf", "Indices Buffer"),
  LongParam("kernSize", "Kernel Size", 3, Min(3), Odd()),
  FloatParam("thresh", "Threshold", 0.8, Min(0.)),
  LongParam("filtSize", "Smoothing Filter Size", 1, Min(1), FrameSizeUpperLimit<kFFT>()),
  FFTParam("fft", "FFT Settings", 1024, -1, -1)
 );

template<typename Params, typename T, typename U>
class NoveltyClient: public FluidBaseClient<Params>, public OfflineIn, public OfflineOut
{

public:

  NoveltyClient(Params& p):FluidBaseClient<Params>{p},mParams{p}
  {}


  Result process()
  {

    if(!param<kSource>(mParams).get())
      return {Result::Status::kError, "No input buffer supplied"};

    BufferAdaptor::Access source(param<kSource>(mParams).get());

    if(!source.exists())
        return {Result::Status::kError, "Input buffer not found"};

    if(!source.valid())
        return {Result::Status::kError, "Can't access input buffer"};


    BufferAdaptor::Access idx(param<kTransBuf>(mParams).get());

    if(!idx.exists())
        return {Result::Status::kError, "Output buffer not found"};

//    if(!idx.valid())
//        return {Result::Status::kError, "Can't access output buffer"};


    auto& fftParams = param<kFFT>(mParams);

    size_t nChannels = param<kNumChans>(mParams)  == -1 ? source.numChans() : param<kNumChans>(mParams);
    size_t nFrames   = param<kNumFrames>(mParams) == -1 ? source.numFrames(): param<kNumFrames>(mParams);
    size_t nWindows  = std::floor((nFrames + fftParams.hopSize()) / fftParams.hopSize());
    size_t nBins     = fftParams.frameSize();

    FluidTensor<double, 1> monoSource(nFrames);

    // Make a mono sum;
    for (size_t i = 0; i < nChannels; ++i) {
      monoSource.apply(
          source.samps(param<kOffset>(mParams), nFrames, param<kStartChan>(mParams) + i),
          [](double &x, double y) { x += y; });
    }


    algorithm::STFT stft(fftParams.winSize(), fftParams.fftSize(), fftParams.hopSize());
    algorithm::ISTFT istft(fftParams.winSize(), fftParams.fftSize(), fftParams.hopSize());

    algorithm::NoveltySegmentation processor(param<kKernelSize>(mParams), param<kThreshold>(mParams),
                                             param<kFilterSize>(mParams));

    auto spectrum = FluidTensor<std::complex<double>,2>(nWindows,nBins);
    auto magnitude = FluidTensor<double,2>(nWindows,nBins);
    auto outputMags = FluidTensor<double,2>(nWindows,nBins);


    stft.process(monoSource, spectrum);
    algorithm::STFT::magnitude(spectrum,magnitude);

    auto changePoints = FluidTensor<double, 1>(magnitude.rows());

    processor.process(magnitude, changePoints);

    impl::spikesToTimes(changePoints(Slice(0)), param<kTransBuf>(mParams).get(), fftParams.hopSize(), param<kOffset>(mParams), nFrames);
    return {Result::Status::kOk,""};
  }

  private:
    Params& mParams;


};
} // namespace client
} // namespace fluid
