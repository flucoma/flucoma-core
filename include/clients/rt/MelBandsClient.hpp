#pragma once

#include "../../algorithms/util/DCT.hpp"
#include "../../algorithms/util/MelBands.hpp"
#include "../../data/TensorTypes.hpp"
#include "../common/AudioClient.hpp"
#include "../common/FluidBaseClient.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/ParameterTypes.hpp"
#include "../nrt/FluidNRTClientWrapper.hpp"
#include "../rt/BufferedProcess.hpp"
#include <clients/common/ParameterTrackChanges.hpp>

namespace fluid {
namespace client {

using algorithm::DCT;
using algorithm::MelBands;

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
  using HostVector = HostVector<T>;

public:
  MelBandsClient(ParamSetViewType &p)
      : FluidBaseClient{p}, mSTFTBufferedProcess(get<kMaxFFTSize>(), 1, 0) {
    mBands = FluidTensor<double, 1>(get<kNBands>());
    FluidBaseClient::audioChannelsIn(1);
    FluidBaseClient::controlChannelsOut(get<kMaxNBands>());
  }

  void process(std::vector<HostVector> &input,
               std::vector<HostVector> &output, bool reset = false) {
    using std::size_t;

    if (!input[0].data() || !output[0].data())
      return;
    assert(FluidBaseClient::controlChannelsOut() && "No control channels");
    assert(output.size() >= FluidBaseClient::controlChannelsOut() &&
           "Too few output channels");

    if (mTracker.changed(get<kFFT>().frameSize(), get<kNBands>(),
                         get<kMinFreq>(), get<kMaxFreq>())) {
      mMagnitude.resize(get<kFFT>().frameSize());
      mBands.resize(get<kNBands>());
      mMelBands.init(get<kMinFreq>(), get<kMaxFreq>(), get<kNBands>(),
                     get<kFFT>().frameSize(), sampleRate(), false);
    }

    mSTFTBufferedProcess.processInput(
        mParams, input, reset, [&](ComplexMatrixView in) {
          algorithm::STFT::magnitude(in.row(0), mMagnitude);
          mMelBands.processFrame(mMagnitude, mBands);
        });
    for (int i = 0; i < get<kNBands>(); ++i)
      output[i](0) = mBands(i);
  }

  size_t latency() { return get<kFFT>().winSize(); }

  size_t controlRate() { return get<kFFT>().hopSize(); }

private:
  ParameterTrackChanges<size_t, size_t, double, double> mTracker;
  STFTBufferedProcess<ParamSetViewType, T, kFFT, false> mSTFTBufferedProcess;
  MelBands mMelBands;
  FluidTensor<double, 1> mMagnitude;
  FluidTensor<double, 1> mBands;
};

auto constexpr NRTMelBandsParams =
    makeNRTParams<MelBandsClient>({BufferParam("source", "Source Buffer")},
                                  {BufferParam("features", "Output Buffer")});

template <typename T>
using NRTMelBandsClient =
    NRTControlAdaptor<MelBandsClient<T>, decltype(NRTMelBandsParams),
                      NRTMelBandsParams, 1, 1>;
} // namespace client
} // namespace fluid
