#pragma once
#include "clients/common/FluidNRTClientWrapper.hpp"
#include "clients/common/ParameterTypes.hpp"
#include "clients/common/ParameterConstraints.hpp"
#include "clients/common/ParameterSet.hpp"
#include "clients/common/FluidBaseClient.hpp"
#include "clients/common/ParameterSet.hpp"
#include "clients/common/ParameterTrackChanges.hpp"
#include "algorithms/NMFCross.hpp"
#include "algorithms/GriffinLim.hpp"
#include "data/FluidIndex.hpp"

namespace fluid {
namespace client {

class NMFCrossClient : public FluidBaseClient, public OfflineIn, public OfflineOut
{

  enum NMFCrossParamIndex{kSource, kTarget, kOutput, kTimeSparsity,  kPolyphony, kIterations, kFFT};

public:

  FLUID_DECLARE_PARAMS(
    InputBufferParam("source", "Source Buffer"),
    InputBufferParam("target", "Target Buffer"),
    BufferParam("output", "Output Buffer"),
    LongParam("timesparsity", "Time sparsity", 10, Min(1), Max(50)),
    LongParam("polyphony", "Polyphony", 7, Min(1), Max(50)),
    LongParam("iterations", "Number of Iterations", 50, Min(1)),
    FFTParam("fftSettings", "FFT Settings", 1024, -1, -1)
  );

  NMFCrossClient(ParamSetViewType& p) : mParams(p)
  {}

  template<typename T>
  Result process(FluidContext&)
  {
    using namespace algorithm;

    auto source = BufferAdaptor::ReadAccess(get<kSource>().get());
    auto target = BufferAdaptor::ReadAccess(get<kTarget>().get());
    BufferAdaptor::Access output(get<kOutput>().get());

    double sampleRate = source.sampleRate();
    auto fftParams = get<kFFT>();

    if(!source.exists())
      return {Result::Status::kError, "Source Buffer Supplied But Invalid"};

    if(!target.exists())
      return {Result::Status::kError, "Target Buffer Supplied But Invalid"};


    if(!output.exists())
      return {Result::Status::kError, "Output Buffer Supplied But Invalid"};


    size_t srcFrames = source.numFrames();
    size_t tgtFrames = target.numFrames();

    if (srcFrames <= 0)
      return {Result::Status::kError, "Empty source buffer"};

    if (tgtFrames <= 0)
      return {Result::Status::kError, "Empty target buffer"};


    size_t srcWindows  = std::floor((srcFrames + fftParams.hopSize()) / fftParams.hopSize());
    size_t nBins     = fftParams.frameSize();

    size_t tgtWindows  = std::floor((tgtFrames + fftParams.hopSize()) / fftParams.hopSize());
    auto stft = STFT(fftParams.winSize(), fftParams.fftSize(), fftParams.hopSize());
    auto istft = ISTFT(fftParams.winSize(), fftParams.fftSize(), fftParams.hopSize());
    auto srcTmp = FluidTensor<double, 1>(srcFrames);
    auto tgtTmp = FluidTensor<double, 1>(tgtFrames);
    auto srcSpectrum = FluidTensor<std::complex<double>,2>(srcWindows,nBins);
    auto tgtSpectrum = FluidTensor<std::complex<double>,2>(tgtWindows,nBins);
    auto W = FluidTensor<double,2>(srcWindows,nBins);
    auto tgtMag = FluidTensor<double,2>(tgtWindows,nBins);

    Result resizeResult = output.resize(tgtFrames, 1,sampleRate);
    if(!resizeResult.ok()) return resizeResult;

    //TODO: addProgressCallback

    //source stft
    srcTmp = source.samps(0, srcFrames, 0);
    stft.process(srcTmp, srcSpectrum);
    STFT::magnitude(srcSpectrum, W);
    // target stft
    tgtTmp = target.samps(0, tgtFrames, 0);
    stft.process(tgtTmp, tgtSpectrum);
    STFT::magnitude(tgtSpectrum, tgtMag);
    index rank = W.rows();
    auto outputEnvelopes = FluidTensor<double, 2>(tgtWindows, rank);
    auto result = FluidTensor<std::complex<double>,2>(tgtWindows,nBins);
    auto final = FluidTensor<std::complex<double>,2>(tgtWindows,nBins);
    auto resultAudio = FluidTensor<double, 1>(tgtFrames);

    auto nmf = NMFCross(get<kIterations>());//update envelopes based on src as W
    //nmf target with source as fixed weights
    nmf.process(tgtMag, outputEnvelopes, W, get<kTimeSparsity>(), get<kPolyphony>());
    NMFCross::synthesize(outputEnvelopes, srcSpectrum, result);
    //improve phase with Griffin-Lim
    GriffinLim gl;
    gl.process(result, tgtFrames, 20,
                 fftParams.winSize(), fftParams.fftSize(), fftParams.hopSize());
    istft.process(result, resultAudio);
    output.samps(0) = resultAudio(Slice(0, tgtFrames));

    return {Result::Status::kOk,""};

  }
};

using NRTNMFCrossClient = NRTThreadingAdaptor<ClientWrapper<NMFCrossClient>>;
} // namespace client
} // namespace fluid
