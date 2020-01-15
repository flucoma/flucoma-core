/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright 2017-2019 University of Huddersfield.
Licensed under the BSD-3 License.
See LICENSE file in the project root for full license information.
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
#include "../../algorithms/util/DCT.hpp"
#include "../../algorithms/util/MelBands.hpp"
#include "../../data/TensorTypes.hpp"

namespace fluid {
namespace client {

enum MFCCParamIndex {
  kNBands,
  kMinFreq,
  kMaxFreq,
  kMaxNBands,
  kFFT,
  kMaxFFTSize
};

auto constexpr MelBandsParams = defineParameters(
    LongParam("numBands", "Number of Bands", 40, Min(2),
              UpperLimit<kMaxNBands>()),
    FloatParam("minFreq", "Low Frequency Bound", 20, Min(0)),
    FloatParam("maxFreq", "High Frequency Bound", 20000, Min(0)),
    LongParam<Fixed<true>>("maxNumBands", "Maximum Number of Bands", 120,
                           Min(2), MaxFrameSizeUpperLimit<kMaxFFTSize>()),
    FFTParam<kMaxFFTSize>("fftSettings", "FFT Settings", 1024, -1, -1),
    LongParam<Fixed<true>>("maxFFTSize", "Maxiumm FFT Size", 16384));

template <typename T>
class MelBandsClient
    : public FluidBaseClient<decltype(MelBandsParams), MelBandsParams>,
      public AudioIn,
      public ControlOut

{
  using HostVector = FluidTensorView<T, 1>;

public:
  MelBandsClient(ParamSetViewType& p)
      : FluidBaseClient{p}, mSTFTBufferedProcess(get<kMaxFFTSize>(), 1, 0)
  {
    mBands = FluidTensor<double, 1>(get<kNBands>());
    FluidBaseClient::audioChannelsIn(1);
    FluidBaseClient::controlChannelsOut(get<kMaxNBands>());
  }

  void process(std::vector<HostVector>& input, std::vector<HostVector>& output,
               FluidContext& c, bool reset = false)
  {
    if (!input[0].data() || !output[0].data()) return;
    assert(FluidBaseClient::controlChannelsOut() && "No control channels");
    assert(output.size() >= asUnsigned(FluidBaseClient::controlChannelsOut()) && "Too few output channels");

    if (mTracker.changed(get<kFFT>().frameSize(), get<kNBands>(),
                         get<kMinFreq>(), get<kMaxFreq>()))
    {
      mMagnitude.resize(get<kFFT>().frameSize());
      mBands.resize(get<kNBands>());
      mMelBands.init(get<kMinFreq>(), get<kMaxFreq>(), get<kNBands>(),
                     get<kFFT>().frameSize(), sampleRate(), false);
    }

    mSTFTBufferedProcess.processInput(
        mParams, input, c, reset, [&](ComplexMatrixView in) {
          algorithm::STFT::magnitude(in.row(0), mMagnitude);
          mMelBands.processFrame(mMagnitude, mBands);
        });
    for (index i = 0; i < get<kNBands>(); ++i) output[asUnsigned(i)](0) = mBands(i);
  }

  index latency() { return get<kFFT>().winSize(); }

  index controlRate() { return get<kFFT>().hopSize(); }

private:
  ParameterTrackChanges<index, index, double, double> mTracker;
  STFTBufferedProcess<ParamSetViewType, T, kFFT, false> mSTFTBufferedProcess;

  algorithm::MelBands mMelBands;

  FluidTensor<double, 1> mMagnitude;
  FluidTensor<double, 1> mBands;
};

auto constexpr NRTMelBandsParams =
    makeNRTParams<MelBandsClient>({InputBufferParam("source", "Source Buffer")},
                                  {BufferParam("features", "Output Buffer")});

template <typename T>
using NRTMelBandsClient =
    NRTControlAdaptor<MelBandsClient<T>, decltype(NRTMelBandsParams),
                      NRTMelBandsParams, 1, 1>;

template <typename T>
using NRTThreadedMelBandsClient = NRTThreadingAdaptor<NRTMelBandsClient<T>>;

} // namespace client
} // namespace fluid
