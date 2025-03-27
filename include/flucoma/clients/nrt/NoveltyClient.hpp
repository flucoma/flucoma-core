/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/

#pragma once

#include "../common/ParameterTypes.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/FluidBaseClient.hpp"
#include "../common/FluidContext.hpp"
#include "../common/SpikesToTimes.hpp"
#include "../../algorithms/public/NoveltySegmentation.hpp"
#include "../../algorithms/public/STFT.hpp"
#include "FluidNRTClientWrapper.hpp"

namespace fluid {
namespace client {

enum NoveltyParamIndex {kSource, kOffset, kNumFrames, kStartChan, kNumChans, kTransBuf, kKernelSize, kThreshold,kFilterSize,kFFT};

auto constexpr NoveltyParams =defineParameters(
  BufferParam("source","Source Buffer"),
  LongParam("startFrame","Source Offset",0,Min(0)),
  LongParam("numFrames","Number of Frames",-1),
  LongParam("startChan","Start Channel",0,Min(0)),
  LongParam("numChans","Number of Channels",-1),
  BufferParam("indices", "Indices Buffer"),
  LongParam("kernelSize", "Kernel Size", 3, Min(3), Odd()),
  FloatParam("threshold", "Threshold", 0.8, Min(0.)),
  LongParam("filterSize", "Smoothing Filter Size", 1, FrameSizeUpperLimit<kFFT>()),
  FFTParam("fftSettings", "FFT Settings", 1024, -1, -1)
 );


class NoveltyClient: public FluidBaseClient<decltype(NoveltyParams), NoveltyParams>, public OfflineIn, public OfflineOut
{

public:

  NoveltyClient(ParamSetViewType& p) : FluidBaseClient(p)
  {}

  template<typename T>
  Result process(FluidContext& c)
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

    impl::spikesToTimes(changePoints(Slice(0)), get<kTransBuf>().get(), fftParams.hopSize(), get<kOffset>(), nFrames, source.sampleRate());
    return {Result::Status::kOk,""};
  }
};

using NRTThreadedNoveltyClient = NRTThreadingAdaptor<NoveltyClient>;

} // namespace client
} // namespace fluid
