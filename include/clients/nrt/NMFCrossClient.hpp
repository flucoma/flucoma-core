#pragma once
#include "CommonResults.hpp"
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
namespace nmfcross {

enum NMFCrossParamIndex {
  kSource,
  kTarget,
  kOutput,
  kTimeSparsity,
  kPolyphony,
  kContinuity,
  kIterations,
  kFFT
};

constexpr auto NMFCrossParams = defineParameters(
    InputBufferParam("source", "Source Buffer"),
    InputBufferParam("target", "Target Buffer"),
    BufferParam("output", "Output Buffer"),
    LongParam("timeSparsity", "Time Sparsity", 7, Min(1), Odd()),
    LongParam("polyphony", "Polyphony", 10, Min(1), Odd(), FrameSizeUpperLimit<kFFT>()),
    LongParam("continuity", "Continuity", 7, Min(1), Odd()),
    LongParam("iterations", "Number of Iterations", 50, Min(1)),
    FFTParam("fftSettings", "FFT Settings", 1024, -1, -1));

class NMFCrossClient : public FluidBaseClient, public OfflineIn, public OfflineOut
{
public:
  using ParamDescType = decltype(NMFCrossParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto getParameterDescriptors() { return NMFCrossParams; }

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

    index srcFrames = source.numFrames();
    index tgtFrames = target.numFrames();

    if (srcFrames <= 0)
      return {Result::Status::kError, "Empty source buffer"};

    if (tgtFrames <= 0)
      return {Result::Status::kError, "Empty target buffer"};

    if(tgtFrames < get<kTimeSparsity>())
      return {Result::Status::kError, "Time Sparsity is larger than target frames"};
    if(tgtFrames < get<kContinuity>())
      return {Result::Status::kError, "Continuity is larger than target frames"};

    index srcWindows  = std::floor((srcFrames + fftParams.hopSize()) / fftParams.hopSize());
    index nBins     = fftParams.frameSize();

    index tgtWindows  = std::floor((tgtFrames + fftParams.hopSize()) / fftParams.hopSize());
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

    srcTmp = source.samps(0, srcFrames, 0);
    stft.process(srcTmp, srcSpectrum);
    STFT::magnitude(srcSpectrum, W);
    tgtTmp = target.samps(0, tgtFrames, 0);
    stft.process(tgtTmp, tgtSpectrum);
    STFT::magnitude(tgtSpectrum, tgtMag);
    index rank = W.rows();
    auto outputEnvelopes = FluidTensor<double, 2>(tgtWindows, rank);
    auto result = FluidTensor<std::complex<double>,2>(tgtWindows, nBins);
    auto final = FluidTensor<std::complex<double>,2>(tgtWindows, nBins);
    auto resultAudio = FluidTensor<double, 1>(tgtFrames);

    auto nmf = NMFCross(get<kIterations>());
    nmf.process(tgtMag, outputEnvelopes, W,
      get<kTimeSparsity>(), get<kPolyphony>(), get<kContinuity>());
    NMFCross::synthesize(outputEnvelopes, srcSpectrum, result);
    GriffinLim gl;
    gl.process(result, tgtFrames, 50,
                 fftParams.winSize(), fftParams.fftSize(), fftParams.hopSize());
    istft.process(result, resultAudio);
    output.samps(0) = resultAudio(Slice(0, tgtFrames));
    return OK();
  }
};
} // namespace nmfcross

using NRTNMFCrossClient =
    NRTThreadingAdaptor<ClientWrapper<nmfcross::NMFCrossClient>>;
} // namespace client
} // namespace fluid
