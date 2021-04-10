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
#include "../../algorithms/public/DCT.hpp"
#include "../../algorithms/public/MelBands.hpp"
#include "../../data/TensorTypes.hpp"

namespace fluid {
namespace client {
namespace mfcc {

enum MFCCParamIndex {
  kNCoefs,
  kNBands,
  kMinFreq,
  kMaxFreq,
  kMaxNCoefs,
  kFFT,
  kMaxFFTSize
};

constexpr auto MFCCParams = defineParameters(
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

class MFCCClient : public FluidBaseClient, public AudioIn, public ControlOut
{
public:
  using ParamDescType = decltype(MFCCParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors() { return MFCCParams; }

  MFCCClient(ParamSetViewType& p)
      : mParams{p}, mSTFTBufferedProcess(get<kMaxFFTSize>(), 1, 0),
        mMelBands(get<kMaxFFTSize>(), get<kMaxFFTSize>()),
        mDCT(get<kMaxFFTSize>(), get<kMaxNCoefs>())
  {
    mBands = FluidTensor<double, 1>(get<kNBands>());
    mCoefficients = FluidTensor<double, 1>(get<kNCoefs>());
    audioChannelsIn(1);
    controlChannelsOut(get<kMaxNCoefs>());
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

    if (mTracker.changed(get<kFFT>().frameSize(), get<kNCoefs>(),
                         get<kNBands>(), get<kMinFreq>(), get<kMaxFreq>(),
                         sampleRate()))
    {
      mMagnitude.resize(get<kFFT>().frameSize());
      mBands.resize(get<kNBands>());
      mCoefficients.resize(get<kNCoefs>());
      mMelBands.init(get<kMinFreq>(), get<kMaxFreq>(), get<kNBands>(),
                     get<kFFT>().frameSize(), sampleRate(),
                     get<kFFT>().winSize());
      mDCT.init(get<kNBands>(), get<kNCoefs>());
    }

    mSTFTBufferedProcess.processInput(
        mParams, input, c, [&](ComplexMatrixView in) {
          algorithm::STFT::magnitude(in.row(0), mMagnitude);
          mMelBands.processFrame(mMagnitude, mBands, false, false, true);
          mDCT.processFrame(mBands, mCoefficients);
        });
    for (index i = 0; i < get<kNCoefs>(); ++i)
      output[asUnsigned(i)](0) = static_cast<T>(mCoefficients(i));
  }

  index latency() { return get<kFFT>().winSize(); }

  void reset()
  {
    mSTFTBufferedProcess.reset();
    mMagnitude.resize(get<kFFT>().frameSize());
    mBands.resize(get<kNBands>());
    mCoefficients.resize(get<kNCoefs>());
    mMelBands.init(get<kMinFreq>(), get<kMaxFreq>(), get<kNBands>(),
                   get<kFFT>().frameSize(), sampleRate(),
                   get<kFFT>().winSize());
    mDCT.init(get<kNBands>(), get<kNCoefs>());
  }

  index controlRate() { return get<kFFT>().hopSize(); }

private:
  ParameterTrackChanges<index, index, index, double, double, double> mTracker;
  STFTBufferedProcess<ParamSetViewType, kFFT, false> mSTFTBufferedProcess;

  algorithm::MelBands    mMelBands;
  algorithm::DCT         mDCT;
  FluidTensor<double, 1> mMagnitude;
  FluidTensor<double, 1> mBands;
  FluidTensor<double, 1> mCoefficients;
};
} // namespace mfcc

using RTMFCCClient = ClientWrapper<mfcc::MFCCClient>;

auto constexpr NRTMFCCParams =
    makeNRTParams<mfcc::MFCCClient>(InputBufferParam("source", "Source Buffer"),
                                    BufferParam("features", "Output Buffer"));

using NRTMFCCClient =
    NRTControlAdaptor<mfcc::MFCCClient, decltype(NRTMFCCParams), NRTMFCCParams,
                      1, 1>;

using NRTThreadedMFCCClient = NRTThreadingAdaptor<NRTMFCCClient>;

} // namespace client
} // namespace fluid
