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
  kDrop0,
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
    LongParam("startCoeff", "Output Coefficient Offset", 0, Min(0),
              Max(1)), // this needs to be programmatically changed to start+num
                       // coeffs <= numBands as discussed
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
        mDCT(get<kMaxFFTSize>(),
             get<kMaxNCoefs>() + 1) // + 1 for possibility of dropping 0th
  {
    mBands = FluidTensor<double, 1>(get<kNBands>());
    mCoefficients = FluidTensor<double, 1>(get<kNCoefs>() + get<kDrop0>());
    audioChannelsIn(1);
    controlChannelsOut({1, get<kMaxNCoefs>()});
    setInputLabels({"audio input"});
    setOutputLabels({"MFCCs"});
  }

  template <typename T>
  void process(std::vector<HostVector<T>>& input,
               std::vector<HostVector<T>>& output, FluidContext& c)
  {
    using std::size_t;

    if (!input[0].data() || !output[0].data()) return;
    assert(controlChannelsOut().count && "No control channels");
    assert(output[0].size() >= controlChannelsOut().size &&
           "Too few output channels");

    bool has0 = !get<kDrop0>();

    if (mTracker.changed(get<kFFT>().frameSize(), get<kNCoefs>() + !has0,
                         get<kNBands>(), get<kMinFreq>(), get<kMaxFreq>(),
                         sampleRate()))
    {
      mMagnitude.resize(get<kFFT>().frameSize());
      mBands.resize(get<kNBands>());
      mCoefficients.resize(get<kNCoefs>() + !has0);
      mMelBands.init(get<kMinFreq>(), get<kMaxFreq>(), get<kNBands>(),
                     get<kFFT>().frameSize(), sampleRate(),
                     get<kFFT>().winSize());
      mDCT.init(get<kNBands>(), get<kNCoefs>() + !has0);
    }

    mSTFTBufferedProcess.processInput(
        mParams, input, c, [&](ComplexMatrixView in) {
          algorithm::STFT::magnitude(in.row(0), mMagnitude);
          mMelBands.processFrame(mMagnitude, mBands, false, false, true);
          mDCT.processFrame(mBands, mCoefficients);
        });
  
      output[0](Slice(0, get<kNCoefs>())) <<=
        mCoefficients(Slice(get<kDrop0>(), get<kNCoefs>()));
      output[0](Slice(get<kNCoefs>(), get<kMaxNCoefs>() - get<kNCoefs>())).fill(0); 
  }

  index latency() { return get<kFFT>().winSize(); }

  void reset()
  {
    mSTFTBufferedProcess.reset();
    mMagnitude.resize(get<kFFT>().frameSize());
    mBands.resize(get<kNBands>());
    mCoefficients.resize(get<kNCoefs>() + get<kDrop0>());
    mMelBands.init(get<kMinFreq>(), get<kMaxFreq>(), get<kNBands>(),
                   get<kFFT>().frameSize(), sampleRate(),
                   get<kFFT>().winSize());
    mDCT.init(get<kNBands>(), get<kNCoefs>() + get<kDrop0>());
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
