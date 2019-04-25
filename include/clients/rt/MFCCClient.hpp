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
  kNCoefs,
  kMinFreq,
  kMaxFreq,
  kOutput,
  kMaxNBands,
  kFFT,
  kMaxFFTSize
};

auto constexpr MFCCParams = defineParameters(
    LongParam("numBands", "Number of Bands", 40, Min(2),
              UpperLimit<kMaxNBands>()),
    LongParam("numCoefs", "Number of Cepstral Coefficients", 13, Min(2),
              UpperLimit<kNBands>()),
    FloatParam("minFreq", "Low Frequency Bound", 20, Min(0)),
    FloatParam("maxFreq", "High Frequency Bound", 20000, Min(0)),
    EnumParam("ouputType", "Output Type", 1, "Bands", "Cepstral Coefficients"),
    LongParam("maxNumBands", "Maximum Number of Bands", 40, Min(2)),
    FFTParam<kMaxFFTSize>("fftSettings", "FFT Settings", 1024, -1, -1),
    LongParam<Fixed<true>>("maxFFTSize", "Maxiumm FFT Size", 16384));

template <typename T>
class MFCCClient : public FluidBaseClient<decltype(MFCCParams), MFCCParams>,
                   public AudioIn,
                   public ControlOut

{
  using HostVector = HostVector<T>;

public:
  MFCCClient(ParamSetViewType &p)
      : FluidBaseClient{p}, mSTFTBufferedProcess(get<kMaxFFTSize>(), 1, 0) {
    mBands = FluidTensor<double, 1>(get<kNBands>());
    mCoefficients = FluidTensor<double, 1>(get<kNCoefs>());
    FluidBaseClient::audioChannelsIn(1);
    FluidBaseClient::controlChannelsOut(get<kMaxNBands>());
  }

  void process(std::vector<HostVector> &input,
               std::vector<HostVector> &output) {
    using std::size_t;

    if (!input[0].data() || !output[0].data())
      return;
    assert(FluidBaseClient::controlChannelsOut() && "No control channels");
    assert(output.size() >= FluidBaseClient::controlChannelsOut() &&
           "Too few output channels");

    if (mTracker.changed(get<kFFT>().frameSize(), get<kNBands>(),
                         get<kNCoefs>(), get<kMinFreq>(), get<kMaxFreq>())) {
      mMagnitude.resize(get<kFFT>().frameSize());
      mBands.resize(get<kNBands>());
      mCoefficients.resize(get<kNCoefs>());
      mMelBands.init(get<kMinFreq>(), get<kMaxFreq>(), get<kNBands>(),
                     get<kFFT>().frameSize(), sampleRate());
      mDCT.init(get<kNBands>(), get<kNCoefs>());
    }

    mSTFTBufferedProcess.processInput(
        mParams, input, [&](ComplexMatrixView in) {
          algorithm::STFT::magnitude(in.row(0), mMagnitude);
          mMelBands.processFrame(mMagnitude, mBands);
          if (get<kOutput>() == 1) {
            mDCT.processFrame(mBands, mCoefficients);
          }
        });
    if (get<kOutput>() == 1) {
      for (int i = 0; i < get<kNCoefs>(); ++i)
        output[i](0) = mCoefficients(i);
    } else {
      for (int i = 0; i < get<kNBands>(); ++i)
        output[i](0) = mBands(i);
    }
  }

  size_t latency() { return get<kFFT>().winSize(); }

  size_t controlRate() { return get<kFFT>().hopSize(); }

private:
  ParameterTrackChanges<size_t, size_t, size_t, double, double> mTracker;
  STFTBufferedProcess<ParamSetViewType, T, kFFT, false> mSTFTBufferedProcess;
  MelBands mMelBands;
  DCT mDCT;
  FluidTensor<double, 1> mMagnitude;
  FluidTensor<double, 1> mBands;
  FluidTensor<double, 1> mCoefficients;
};

auto constexpr NRTMFCCParams =
    makeNRTParams<MFCCClient>({BufferParam("source", "Source Buffer")},
                              {BufferParam("features", "Output Buffer")});

template <typename T>
using NRTMFCCClient = NRTControlAdaptor<MFCCClient<T>, decltype(NRTMFCCParams),
                                        NRTMFCCParams, 1, 1>;

} // namespace client
} // namespace fluid
