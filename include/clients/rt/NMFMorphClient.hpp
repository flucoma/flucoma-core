/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright 2017-2019 University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/
#pragma once

#include "../../algorithms/NMFMorph.hpp"
#include "clients/common/BufferedProcess.hpp"
#include "clients/common/FluidBaseClient.hpp"
#include "clients/common/ParameterConstraints.hpp"
#include "clients/common/ParameterSet.hpp"
#include "clients/common/ParameterTrackChanges.hpp"
#include "clients/common/ParameterTypes.hpp"

namespace fluid {
namespace client {
namespace nmfmorph {

enum NMFFilterIndex {
  kSourceBuf,
  kTargetBuf,
  kActBuf,
  kAutoAssign,
  kInterp,
  kFFT,
  kMaxFFTSize
};

constexpr auto NMFMorphParams = defineParameters(
    InputBufferParam("source", "Source Bases"),
    InputBufferParam("target", "Target Bases"),
    InputBufferParam("activations", "Activations"),
    EnumParam("autoassign", "Automatic assign", 1, "No", "Yes"),
    FloatParam("interp", "Interpolation", 0, Min(0.0), Max(1.0)),
    FFTParam<kMaxFFTSize>("fftSettings", "FFT Settings", 1024, -1, -1),
    LongParam<Fixed<true>>("maxFFTSize", "Maxiumm FFT Size", 16384, Min(4),
                           PowerOfTwo{}));

class NMFMorphClient : public FluidBaseClient, public AudioOut {

public:
  using ParamDescType = decltype(NMFMorphParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto getParameterDescriptors() { return NMFMorphParams; }

  NMFMorphClient(ParamSetViewType &p)
      : mParams{p}, mSTFTProcessor{get<kMaxFFTSize>(), 0, 1} {
    audioChannelsIn(0);
    audioChannelsOut(1);
  }

  index latency() { return get<kFFT>().winSize(); }

  void reset() { mSTFTProcessor.reset(); }

  template <typename T>
  void process(std::vector<HostVector<T>> &,
               std::vector<HostVector<T>> &output, FluidContext &c) {
    assert(audioChannelsOut() && "No control channels");
    assert(output.size() >= asUnsigned(audioChannelsOut()) &&
           "Too few output channels");
    if (get<kSourceBuf>().get() && get<kTargetBuf>().get() &&
        get<kActBuf>().get()) {
      auto &fftParams = get<kFFT>();
      auto sourceBuffer = BufferAdaptor::ReadAccess(get<kSourceBuf>().get());
      auto targetBuffer = BufferAdaptor::ReadAccess(get<kTargetBuf>().get());
      auto actBuffer = BufferAdaptor::ReadAccess(get<kActBuf>().get());
      if (!sourceBuffer.valid() || !targetBuffer.valid() ||
          !actBuffer.valid()) {
        return;
      }
      index rank = sourceBuffer.numChans();
      if (targetBuffer.numChans() != rank || actBuffer.numChans() != rank)
        return;
      if (sourceBuffer.numFrames() != fftParams.frameSize()) {
        return;
      }
      if (sourceBuffer.numFrames() != targetBuffer.numFrames()) {
        return;
      }
      if (mTrackValues.changed(rank, fftParams.frameSize(),get<kAutoAssign>())) {
        tmpSource.resize(rank, fftParams.frameSize());
        tmpTarget.resize(rank, fftParams.frameSize());
        tmpAct.resize(rank, actBuffer.numFrames());
        for (index i = 0; i < rank; ++i) {
          tmpSource.row(i) = sourceBuffer.samps(i);
          tmpTarget.row(i) = targetBuffer.samps(i);
          tmpAct.row(i) = actBuffer.samps(i);
        }
        mNMFMorph.init(tmpSource, tmpTarget, tmpAct, fftParams.winSize(),
                       fftParams.fftSize(), fftParams.hopSize(), get<kAutoAssign>() == 1);
      }
      mSTFTProcessor.processOutput(
          mParams, output, c,
          [&](ComplexMatrixView out) { mNMFMorph.processFrame(out.row(0), get<kInterp>()); });
    }
  }

private:
  ParameterTrackChanges<index, index, index> mTrackValues;
  STFTBufferedProcess<ParamSetViewType, kFFT, true> mSTFTProcessor;
  algorithm::NMFMorph mNMFMorph;
  RealMatrix tmpSource;
  RealMatrix tmpTarget;
  RealMatrix tmpAct;
};
} // namespace nmfmorph

using RTNMFMorphClient = ClientWrapper<nmfmorph::NMFMorphClient>;

} // namespace client
} // namespace fluid
