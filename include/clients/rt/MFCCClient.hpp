/*
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
              UpperLimit<kNBands, kMaxNCoefs>()),
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
  using HostVector = FluidTensorView<T, 1>;

public:
  MFCCClient(ParamSetViewType& p)
      : FluidBaseClient{p}, mSTFTBufferedProcess(get<kMaxFFTSize>(), 1, 0)
  {
    mBands = FluidTensor<double, 1>(get<kNBands>());
    mCoefficients = FluidTensor<double, 1>(get<kNCoefs>());
    FluidBaseClient::audioChannelsIn(1);
    FluidBaseClient::controlChannelsOut(get<kMaxNCoefs>());
  }

  void process(std::vector<HostVector>& input, std::vector<HostVector>& output,
               FluidContext& c, bool reset = false)
  {
    if (!input[0].data() || !output[0].data()) return;
    assert(FluidBaseClient::controlChannelsOut() && "No control channels");
    assert(output.size() >= asUnsigned(FluidBaseClient::controlChannelsOut()) &&
           "Too few output channels");

    if (mTracker.changed(get<kFFT>().frameSize(), get<kNCoefs>(),
                         get<kNBands>(), get<kMinFreq>(), get<kMaxFreq>()))
    {
      mMagnitude.resize(get<kFFT>().frameSize());
      mBands.resize(get<kNBands>());
      mCoefficients.resize(get<kNCoefs>());
      mMelBands.init(get<kMinFreq>(), get<kMaxFreq>(), get<kNBands>(),
                     get<kFFT>().frameSize(), sampleRate(), true);
      mDCT.init(get<kNBands>(), get<kNCoefs>());
    }

    mSTFTBufferedProcess.processInput(
        mParams, input, c, reset, [&](ComplexMatrixView in) {
          algorithm::STFT::magnitude(in.row(0), mMagnitude);
          mMelBands.processFrame(mMagnitude, mBands);
          mDCT.processFrame(mBands, mCoefficients);
        });
    for (index i = 0; i < get<kNCoefs>(); ++i) output[asUnsigned(i)](0) = mCoefficients(i);
  }

  index latency() { return get<kFFT>().winSize(); }

  index controlRate() { return get<kFFT>().hopSize(); }

private:
  ParameterTrackChanges<index, index, index, double, double> mTracker;
  STFTBufferedProcess<ParamSetViewType, T, kFFT, false> mSTFTBufferedProcess;

  algorithm::MelBands mMelBands;
  algorithm::DCT      mDCT;

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

template <typename T>
using NRTThreadedMFCCClient = NRTThreadingAdaptor<NRTMFCCClient<T>>;

} // namespace client
} // namespace fluid
