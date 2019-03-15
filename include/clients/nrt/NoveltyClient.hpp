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
  LongParam("filtSize", "Smoothing Filter Size", 1, FrameSizeUpperLimit<kFFT>()),
  FFTParam("fft", "FFT Settings", 1024, -1, -1)
 );

template<typename T>
class NoveltyClient: public FluidBaseClient<decltype(NoveltyParams), NoveltyParams>, public OfflineIn, public OfflineOut
{

public:

  NoveltyClient(ParamSetType& p) : FluidBaseClient(p)
  {}


  Result process()
  {

    if(!get<kSource>().get())
      return {Result::Status::kError, "No input buffer supplied"};

    BufferAdaptor::Access source(get<kSource>().get());

    if(!source.exists())
        return {Result::Status::kError, "Input buffer not found"};

    if(!source.valid())
        return {Result::Status::kError, "Can't access input buffer"};


    {
    BufferAdaptor::Access idx(get<kTransBuf>().get());

    if(!idx.exists())
        return {Result::Status::kError, "Output buffer not found"};
    
    }

//    if(!idx.valid())
//        return {Result::Status::kError, "Can't access output buffer"};


    auto& fftParams = get<kFFT>();

    size_t nChannels = get<kNumChans>()  == -1 ? source.numChans() : get<kNumChans>();
    size_t nFrames   = get<kNumFrames>() == -1 ? source.numFrames(): get<kNumFrames>();
    size_t nWindows  = std::floor((nFrames + fftParams.hopSize()) / fftParams.hopSize());
    size_t nBins     = fftParams.frameSize();

    FluidTensor<double, 1> monoSource(nFrames);

    // Make a mono sum;
    for (size_t i = 0; i < nChannels; ++i) {
      monoSource.apply(
          source.samps(get<kOffset>(), nFrames, get<kStartChan>() + i),
          [](double &x, double y) { x += y; });
    }


    algorithm::STFT stft(fftParams.winSize(), fftParams.fftSize(), fftParams.hopSize());
    algorithm::ISTFT istft(fftParams.winSize(), fftParams.fftSize(), fftParams.hopSize());

    algorithm::NoveltySegmentation processor(get<kKernelSize>(), get<kThreshold>(),
                                             get<kFilterSize>());

    auto spectrum = FluidTensor<std::complex<double>,2>(nWindows,nBins);
    auto magnitude = FluidTensor<double,2>(nWindows,nBins);
    auto outputMags = FluidTensor<double,2>(nWindows,nBins);

    stft.process(monoSource, spectrum);
    algorithm::STFT::magnitude(spectrum,magnitude);

    auto changePoints = FluidTensor<double, 1>(magnitude.rows());

    processor.process(magnitude, changePoints);

    impl::spikesToTimes(changePoints(Slice(0)), get<kTransBuf>().get(), fftParams.hopSize(), get<kOffset>(), nFrames);
    return {Result::Status::kOk,""};
  }
};
} // namespace client
} // namespace fluid
