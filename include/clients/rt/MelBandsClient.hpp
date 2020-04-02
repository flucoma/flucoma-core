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

#include "../common/AudioClient.hpp"
#include "../common/BufferedProcess.hpp"
#include "../common/FluidBaseClient.hpp"
#include "../common/FluidNRTClientWrapper.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/ParameterTrackChanges.hpp"
#include "../common/ParameterTypes.hpp"
#include "../../algorithms/public/MelBands.hpp"
#include "../../data/TensorTypes.hpp"

namespace fluid {
namespace client {


class MelBandsClient : public FluidBaseClient, public AudioIn, public ControlOut
{
  enum MFCCParamIndex {
    kNBands,
    kMinFreq,
    kMaxFreq,
    kMaxNBands,
    kNormalize,
    kFFT,
    kMaxFFTSize
  };

public:
  FLUID_DECLARE_PARAMS(
      LongParam("numBands", "Number of Bands", 40, Min(2),
                UpperLimit<kMaxNBands>()),
      FloatParam("minFreq", "Low Frequency Bound", 20, Min(0)),
      FloatParam("maxFreq", "High Frequency Bound", 20000, Min(0)),
      LongParam<Fixed<true>>("maxNumBands", "Maximum Number of Bands", 120,
                             Min(2), MaxFrameSizeUpperLimit<kMaxFFTSize>()),
      EnumParam("normalize", "Normalize", 1, "No", "Yes"),
      FFTParam<kMaxFFTSize>("fftSettings", "FFT Settings", 1024, -1, -1),
      LongParam<Fixed<true>>("maxFFTSize", "Maxiumm FFT Size", 16384));

  MelBandsClient(ParamSetViewType& p)
      : mParams{p}, mSTFTBufferedProcess(get<kMaxFFTSize>(), 1, 0),
        mMelBands(get<kMaxNBands>(), get<kMaxFFTSize>())
  {
    mBands = FluidTensor<double, 1>(get<kNBands>());
    audioChannelsIn(1);
    controlChannelsOut(get<kMaxNBands>());
  }

  template <typename T>
  void process(std::vector<HostVector<T>>& input,
               std::vector<HostVector<T>>& output, FluidContext& c)
  {
    using std::size_t;

    if (!input[0].data() || !output[0].data()) return;
    assert(controlChannelsOut() && "No control channels");
    assert(output.size() >= asUnsigned(controlChannelsOut()) &&
           "Too few output channels");
    if (mTracker.changed(get<kFFT>().winSize(), get<kFFT>().frameSize(),
                         get<kNBands>(), get<kNormalize>(), get<kMinFreq>(),
                         get<kMaxFreq>(), sampleRate()))
    {
      mMagnitude.resize(get<kFFT>().frameSize());
      mBands.resize(get<kNBands>());
      mMelBands.init(get<kMinFreq>(), get<kMaxFreq>(), get<kNBands>(),
                     get<kFFT>().frameSize(), sampleRate(),
                     get<kFFT>().winSize());
    }

    mSTFTBufferedProcess.processInput(
        mParams, input, c, [&](ComplexMatrixView in) {
          algorithm::STFT::magnitude(in.row(0), mMagnitude);
          mMelBands.processFrame(mMagnitude, mBands, get<kNormalize>() == 1,
                                 false, false);
        });
    for (index i = 0; i < get<kNBands>(); ++i)
      output[asUnsigned(i)](0) = static_cast<T>(mBands(i));
  }

  index latency() { return get<kFFT>().winSize(); }

  void reset()
  {
    mSTFTBufferedProcess.reset();
    mMelBands.init(get<kMinFreq>(), get<kMaxFreq>(), get<kNBands>(),
                   get<kFFT>().frameSize(), sampleRate(),
                   get<kFFT>().winSize());
  }

  index controlRate() { return get<kFFT>().hopSize(); }

private:
  ParameterTrackChanges<index, index, index, index, double, double, double>
                                                     mTracker;
  STFTBufferedProcess<ParamSetViewType, kFFT, false> mSTFTBufferedProcess;

  algorithm::MelBands    mMelBands;
  FluidTensor<double, 1> mMagnitude;
  FluidTensor<double, 1> mBands;
};

using RTMelBandsClient = ClientWrapper<MelBandsClient>;

auto constexpr NRTMelBandsParams = makeNRTParams<MelBandsClient>(
    InputBufferParam("source", "Source Buffer"),
    BufferParam("features", "Output Buffer"));

using NRTMelBandsClient =
    NRTControlAdaptor<MelBandsClient, decltype(NRTMelBandsParams),
                      NRTMelBandsParams, 1, 1>;

using NRTThreadedMelBandsClient = NRTThreadingAdaptor<NRTMelBandsClient>;

} // namespace client
} // namespace fluid
