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

#include "../../algorithms/public/NMFMorph.hpp"
#include "../../algorithms/public/NMF.hpp"
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
  kAutoAssign,
  kInterp,
  kFFT,
  kMaxFFTSize
};

constexpr auto NMFMorphParams = defineParameters(
    InputBufferParam("source", "Source Bases"),
    InputBufferParam("target", "Target Bases"),
    EnumParam("autoassign", "Automatic assign", 1, "No", "Yes"),
    FloatParam("interp", "Interpolation", 0, Min(0.0), Max(1.0)),
    FFTParam<kMaxFFTSize>("fftSettings", "FFT Settings", 1024, -1, -1),
    LongParam<Fixed<true>>("maxFFTSize", "Maxiumm FFT Size", 16384, Min(4),
                           PowerOfTwo{}));

class NMFMorphInClient : public FluidBaseClient, public AudioOut {

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

  NMFMorphInClient(ParamSetViewType &p)
      : mParams{p}, mSTFTProcessor{get<kMaxFFTSize>(), 1, 1} {
    audioChannelsIn(1);
    audioChannelsOut(1);
  }

  index latency() { return get<kFFT>().winSize(); }

  void reset() { mSTFTProcessor.reset(); }

  template <typename T>
  void process(std::vector<HostVector<T>> &input,
               std::vector<HostVector<T>> &output, FluidContext &c) {
    assert(audioChannelsOut() && "No control channels");
    assert(output.size() >= asUnsigned(audioChannelsOut()) &&
           "Too few output channels");
    if (get<kSourceBuf>().get() && get<kTargetBuf>().get()) {

      auto &fftParams = get<kFFT>();
      auto sourceBuffer = BufferAdaptor::ReadAccess(get<kSourceBuf>().get());
      auto targetBuffer = BufferAdaptor::ReadAccess(get<kTargetBuf>().get());
      if (!sourceBuffer.valid() || !targetBuffer.valid()) {
        return;
      }       

      index rank = sourceBuffer.numChans();
      if (targetBuffer.numChans() != rank)
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
        tmpAct.resize(rank);
        tmpActMatrix.resize(rank, 1);
        tmpMagnitude.resize(1, fftParams.frameSize());
        for (index i = 0; i < rank; ++i) {
          tmpSource.row(i) = sourceBuffer.samps(i);
          tmpTarget.row(i) = targetBuffer.samps(i);
        }
        mNMFMorph.init(tmpSource, tmpTarget, tmpActMatrix, fftParams.winSize(),
                       fftParams.fftSize(), fftParams.hopSize(), get<kAutoAssign>() == 1);
      }
      
      mSTFTProcessor.process(mParams, input, output, c, [&](ComplexMatrixView in, ComplexMatrixView out) {
        algorithm::STFT::magnitude(in, tmpMagnitude);
        mNMF.processFrame(tmpMagnitude.row(0), tmpSource, tmpAct);
        mNMFMorph.processFrame(out.row(0), tmpAct, get<kInterp>());
      });
    }
  }

private:
  ParameterTrackChanges<index, index, index> mTrackValues;
  STFTBufferedProcess<ParamSetViewType, kFFT, true> mSTFTProcessor;
  algorithm::NMFMorph mNMFMorph;
  algorithm::NMF mNMF;
  RealMatrix tmpSource;
  RealMatrix tmpTarget;
  RealMatrix tmpActMatrix;
  RealMatrix    tmpMagnitude;
  RealVector   tmpAct;
};
} // namespace nmfmorph

using RTNMFMorphInClient = ClientWrapper<nmfmorph::NMFMorphInClient>;

} // namespace client
} // namespace fluid
