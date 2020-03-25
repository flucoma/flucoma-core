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
#include "../common/ParameterTypes.hpp"
#include "../../algorithms/public/CepstrumF0.hpp"
#include "../../algorithms/public/HPS.hpp"
#include "../../algorithms/public/YINFFT.hpp"
#include "../../data/TensorTypes.hpp"
#include <tuple>

namespace fluid {
namespace client {

enum PitchParamIndex {
  kAlgorithm,
  kMinFreq,
  kMaxFreq,
  kUnit,
  kFFT,
  kMaxFFTSize
};

extern auto constexpr PitchParams = defineParameters(
    EnumParam("algorithm", "Algorithm", 2, "Cepstrum",
              "Harmonic Product Spectrum", "YinFFT"),
    FloatParam("minFreq", "Minimum Frequency", 20, Min(0), Max(10000),
               UpperLimit<kMaxFreq>()),
    FloatParam("maxFreq", "Maximum Frequency", 10000, Min(1), Max(20000),
               LowerLimit<kMinFreq>()),
    EnumParam("unit", "Unit", 0, "Hz", "MIDI"),
    FFTParam<kMaxFFTSize>("fftSettings", "FFT Settings", 1024, -1, -1),
    LongParam<Fixed<true>>("maxFFTSize", "Maxiumm FFT Size", 16384, Min(4),
                           PowerOfTwo{}));

template <typename T>
class PitchClient : public FluidBaseClient<decltype(PitchParams), PitchParams>,
                    public AudioIn,
                    public ControlOut
{
  using HostVector = FluidTensorView<T, 1>;
  using CepstrumF0 = algorithm::CepstrumF0;
  using HPS = algorithm::HPS;
  using YINFFT = algorithm::YINFFT;

public:
  PitchClient(ParamSetViewType& p)
      : FluidBaseClient(p), mSTFTBufferedProcess(get<kMaxFFTSize>(), 1, 0)
  {
    FluidBaseClient::audioChannelsIn(1);
    FluidBaseClient::controlChannelsOut(2);
    mDescriptors = FluidTensor<double, 1>(2);
  }

  void process(std::vector<HostVector>& input, std::vector<HostVector>& output,
               FluidContext& c)
  {
    if (!input[0].data() || !output[0].data()) return;
    assert(FluidBaseClient::controlChannelsOut() && "No control channels");
    assert(asSigned(output.size()) >= FluidBaseClient::controlChannelsOut() &&
           "Too few output channels");

    if (mParamTracker.changed(get<kFFT>().frameSize(), sampleRate()))
    {
      cepstrumF0.init(get<kFFT>().frameSize());
      mMagnitude.resize(get<kFFT>().frameSize());
    }

    mSTFTBufferedProcess.processInput(
        mParams, input, c, [&](ComplexMatrixView in) {
          algorithm::STFT::magnitude(in.row(0), mMagnitude);
          switch (get<kAlgorithm>())
          {
          case 0:
            cepstrumF0.processFrame(mMagnitude, mDescriptors, get<kMinFreq>(),
                                    get<kMaxFreq>(), sampleRate());
            break;
          case 1:
            hps.processFrame(mMagnitude, mDescriptors, 4, get<kMinFreq>(),
                             get<kMaxFreq>(), sampleRate());
            break;
          case 2:
            yinFFT.processFrame(mMagnitude, mDescriptors, get<kMinFreq>(),
                                get<kMaxFreq>(), sampleRate());
            break;
          }
        });
    output[0](0) = static_cast<T>(
        get<kUnit>() == 0 ? mDescriptors(0)
                          : 69 + (12 * log2(mDescriptors(0) / 440.0))); // pitch
    output[1](0) = static_cast<T>(mDescriptors(1)); // pitch confidence
  }
  index latency() { return get<kFFT>().winSize(); }
  index controlRate() { return get<kFFT>().hopSize(); }
  void  reset() { mSTFTBufferedProcess.reset(); }

private:
  ParameterTrackChanges<index, double>           mParamTracker;
  STFTBufferedProcess<ParamSetViewType, T, kFFT> mSTFTBufferedProcess;

  CepstrumF0             cepstrumF0{get<kMaxFFTSize>()};
  HPS                    hps;
  YINFFT                 yinFFT;
  FluidTensor<double, 1> mMagnitude;
  FluidTensor<double, 1> mDescriptors;
};

auto constexpr NRTPitchParams =
    makeNRTParams<PitchClient>(InputBufferParam("source", "Source Buffer"),
                               BufferParam("features", "Features Buffer"));

template <typename T>
using NRTPitchClient =
    NRTControlAdaptor<PitchClient<T>, decltype(NRTPitchParams), NRTPitchParams,
                      1, 1>;

template <typename T>
using NRTThreadedPitchClient = NRTThreadingAdaptor<NRTPitchClient<T>>;

} // namespace client
} // namespace fluid
