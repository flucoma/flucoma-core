#pragma once

#include "BufferedProcess.hpp"
#include "../common/AudioClient.hpp"
#include "../common/FluidBaseClient.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/ParameterTypes.hpp"
#include "../common/ParameterTrackChanges.hpp"
#include "../nrt/FluidNRTClientWrapper.hpp"
#include "../../algorithms/util/DCT.hpp"
#include "../../algorithms/util/MelBands.hpp"
#include "../../data/TensorTypes.hpp"

namespace fluid {
namespace client {

using algorithm::DCT;
using algorithm::MelBands;

enum MFCCParamIndex {
  kNCoefs,
  kNBands,
  kMinFreq,
  kMaxFreq,
  kMaxNCoefs,
  kFFT,
  kMaxFFTSize
};

auto constexpr MFCCParams = defineParameters(

    LongParam("numCoeffs", "Number of Cepstral Coefficients", 13, Min(2),
              UpperLimit<kNBands,kMaxNCoefs>()),
    LongParam("numBands", "Number of Bands", 40, Min(2),
              FrameSizeUpperLimit<kFFT>(), LowerLimit<kNCoefs>()),
    FloatParam("minFreq", "Low Frequency Bound", 20, Min(0)),
    FloatParam("maxFreq", "High Frequency Bound", 20000, Min(0)),
    LongParam<Fixed<true>>("maxNumCoeffs", "Maximum Number of Coefficients", 40,
                           MaxFrameSizeUpperLimit<kMaxFFTSize>(), Min(2)),
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
    FluidBaseClient::controlChannelsOut(get<kMaxNCoefs>());
  }

  void process(std::vector<HostVector> &input,
               std::vector<HostVector> &output, bool reset = false) {
    using std::size_t;

    if (!input[0].data() || !output[0].data())
      return;
    assert(FluidBaseClient::controlChannelsOut() && "No control channels");
    assert(output.size() >= FluidBaseClient::controlChannelsOut() &&
           "Too few output channels");

    if (mTracker.changed(get<kFFT>().frameSize(), get<kNCoefs>(),
                         get<kNBands>(), get<kMinFreq>(), get<kMaxFreq>())) {
      mMagnitude.resize(get<kFFT>().frameSize());
      mBands.resize(get<kNBands>());
      mCoefficients.resize(get<kNCoefs>());
      mMelBands.init(get<kMinFreq>(), get<kMaxFreq>(), get<kNBands>(),
                     get<kFFT>().frameSize(), sampleRate(), true);
      mDCT.init(get<kNBands>(), get<kNCoefs>());
    }

    mSTFTBufferedProcess.processInput(
        mParams, input, reset, [&](ComplexMatrixView in) {
          algorithm::STFT::magnitude(in.row(0), mMagnitude);
          mMelBands.processFrame(mMagnitude, mBands);
          mDCT.processFrame(mBands, mCoefficients);
        });
    for (int i = 0; i < get<kNCoefs>(); ++i)
      output[i](0) = mCoefficients(i);
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
    makeNRTParams<MFCCClient>({InputBufferParam("source", "Source Buffer")},
                              {BufferParam("features", "Output Buffer")});
template <typename T>
using NRTMFCCClient = NRTControlAdaptor<MFCCClient<T>, decltype(NRTMFCCParams),
                                        NRTMFCCParams, 1, 1>;

} // namespace client
} // namespace fluid
