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

#include "../common/BufferedProcess.hpp"
#include "../common/FluidBaseClient.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/ParameterTrackChanges.hpp"
#include "../common/ParameterTypes.hpp"
#include "../../algorithms/public/NMF.hpp"

namespace fluid {
namespace client {

enum NMFMatchParamIndex {
  kFilterbuf,
  kMaxRank,
  kIterations,
  kFFT,
  kMaxFFTSize
};

extern auto constexpr NMFMatchParams = defineParameters(
    InputBufferParam("bases", "Bases Buffer"),
    LongParam<Fixed<true>>("maxComponents", "Maximum Number of Components", 20,
                           Min(1)),
    LongParam("iterations", "Number of Iterations", 10, Min(1)),
    FFTParam<kMaxFFTSize>("fftSettings", "FFT Settings", 1024, -1, -1),
    LongParam<Fixed<true>>("maxFFTSize", "Maxiumm FFT Size", 16384, Min(4),
                           PowerOfTwo{}));


template <typename T>
class NMFMatchClient
    : public FluidBaseClient<decltype(NMFMatchParams), NMFMatchParams>,
      public AudioIn,
      public ControlOut
{
  using HostVector = FluidTensorView<T, 1>;

public:
  NMFMatchClient(ParamSetViewType& p)
      : FluidBaseClient(p), mSTFTProcessor(get<kMaxFFTSize>(), 1, 0)
  {
    audioChannelsIn(1);
    controlChannelsOut(get<kMaxRank>());
  }

  index latency() { return get<kFFT>().winSize(); }

  void reset() { mSTFTProcessor.reset(); }

  void process(std::vector<HostVector>& input, std::vector<HostVector>& output,
               FluidContext& c)
  {
    if (!input[0].data()) return;
    assert(FluidBaseClient::controlChannelsOut() && "No control channels");
    assert(output.size() >= asUnsigned(FluidBaseClient::controlChannelsOut()) &&
           "Too few output channels");

    if (get<kFilterbuf>().get())
    {

      auto  filterBuffer = BufferAdaptor::ReadAccess(get<kFilterbuf>().get());
      auto& fftParams = get<kFFT>();

      if (!filterBuffer.valid()) { return; }

      index rank = std::min<index>(filterBuffer.numChans(), get<kMaxRank>());

      if (filterBuffer.numFrames() != fftParams.frameSize()) { return; }

      if (mTrackValues.changed(rank, fftParams.frameSize()))
      {
        tmpFilt.resize(rank, fftParams.frameSize());
        tmpMagnitude.resize(1, fftParams.frameSize());
        tmpOut.resize(rank);
      }

      for (index i = 0; i < tmpFilt.rows(); ++i)
        tmpFilt.row(i) = filterBuffer.samps(i);

      //      controlTrigger(false);
      mSTFTProcessor.processInput(mParams, input, c, [&](ComplexMatrixView in) {
        algorithm::STFT::magnitude(in, tmpMagnitude);
        mNMF.processFrame(tmpMagnitude.row(0), tmpFilt, tmpOut);
        //          controlTrigger(true);
      });

      for (index i = 0; i < rank; ++i)
        output[asUnsigned(i)](0) = static_cast<T>(tmpOut(i));
    }
  }

private:
  ParameterTrackChanges<index, index> mTrackValues;
  algorithm::NMF                      mNMF;
  FluidTensor<double, 2>              tmpFilt;
  FluidTensor<double, 2>              tmpMagnitude;
  FluidTensor<double, 1>              tmpOut;

  STFTBufferedProcess<ParamSetViewType, T, kFFT, false> mSTFTProcessor;

  size_t mNBins{0};
  size_t mRank{0};
};
} // namespace client
} // namespace fluid
