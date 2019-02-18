#pragma once


#include <clients/common/ParameterTypes.hpp>
#include <clients/common/ParameterConstraints.hpp>
#include <clients/common/FluidBaseClient.hpp>
#include <clients/common/DeriveSTFTParams.hpp>
#include <clients/common/SpikesToTimes.hpp>
#include <algorithms/public/NoveltySegmentation.hpp>
#include <algorithms/public/STFT.hpp>

namespace fluid {
namespace client {

enum NoveltyParamIndex {kTransBuf, kKernelSize, kThreshold,kFilterSize,kWinSize, kHopSize, kFFTSize, kMaxWinSize}; 

auto constexpr NoveltyParams = std::make_tuple(
  BufferParam("transBuf", "Indices Buffer"),
  LongParam("kernelSize", "Kernel Size", 3, Min(3), Odd()),
  FloatParam("threshold", "Threshold", 0.8, Min(0.)),
  LongParam("filterSize", "Smoothing Filter Size",256, Min(3), FrameSizeUpperLimit<kWinSize,kFFTSize>()),
  LongParam("winSize", "Window Size",1024, Min(4), UpperLimit<kFFTSize>()),
  LongParam("hopSize", "Hop Size",512),
  LongParam("fftSize", "FFT Size", 2048,LowerLimit<kWinSize>(),PowerOfTwo()),
  LongParam("maxWinSize", "Maxiumm Window Size", 16384));


using ParamsT = decltype(NoveltyParams);

/**
 Integration class for doing NMF filtering and resynthesis
 **/

class NoveltyClient: public FluidBaseClient<ParamsT>, public OfflineIn, public OfflineOut
{

public:

  NoveltyClient():FluidBaseClient<ParamsT>(NoveltyParams)
  {
    audioBuffersIn(1);
    audioBuffersOut(0);
  }

   Result process(std::vector<BufferProcessSpec>& inputs, std::vector<BufferProcessSpec>&)
  {
  
    if(!inputs[0].buffer)
      return {Result::Status::kError, "No input buffer supplied"};
    
    BufferAdaptor::Access source(inputs[0].buffer);
    
    if(!source.exists())
        return {Result::Status::kError, "Input buffer not found"};
    
    if(!source.valid())
        return {Result::Status::kError, "Can't access input buffer"};

    
    BufferAdaptor::Access idx(get<kTransBuf>().get());

    if(!idx.exists())
        return {Result::Status::kError, "Output buffer not found"};
    
    if(!idx.valid())
        return {Result::Status::kError, "Can't access output buffer"};

    
    size_t winSize, hopSize, fftSize;
    
    std::tie(winSize,hopSize,fftSize) = impl::deriveSTFTParams<kWinSize, kHopSize, kFFTSize>(*this);

    size_t nChannels = inputs[0].nChans == -1 ? source.numChans() : inputs[0].nChans;
    size_t nFrames   = inputs[0].nFrames == -1  ? source.numFrames(): inputs[0].nFrames;
    size_t nWindows  = std::floor((nFrames + hopSize) / hopSize);
    size_t nBins     = fftSize / 2 + 1;
    
    FluidTensor<double, 1> monoSource(nFrames);

    // Make a mono sum;
    for (size_t i = 0; i < nChannels; ++i) {
      monoSource.apply(
          source.samps(inputs[0].startFrame, nFrames, inputs[0].startChan + i),
          [](double &x, double y) { x += y; });
    }
    
  
    algorithm::STFT stft(winSize, fftSize, hopSize);
    algorithm::ISTFT istft(winSize, fftSize, hopSize);

    algorithm::NoveltySegmentation processor(get<kKernelSize>(), get<kThreshold>(),
                                             get<kFilterSize>());

    auto spectrum = FluidTensor<std::complex<double>,2>(nWindows,nBins);
    auto magnitude = FluidTensor<double,2>(nWindows,nBins);
    auto outputMags = FluidTensor<double,2>(nWindows,nBins);


    stft.process(monoSource, spectrum);
    algorithm::STFT::magnitude(spectrum,magnitude);
    
    auto changePoints = FluidTensor<double, 1>(magnitude.rows());
    
    processor.process(magnitude, changePoints);
    
    impl::spikesToTimes(changePoints(Slice(0)), get<kTransBuf>().get(), hopSize, inputs[0].startFrame, nFrames);
    return {Result::Status::kOk,""}; 
  }
};
} // namespace client
} // namespace fluid
