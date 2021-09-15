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
namespace pitch {

enum PitchParamIndex {
  kAlgorithm,
  kMinFreq,
  kMaxFreq,
  kUnit,
  kFFT,
  kMaxFFTSize
};

constexpr auto PitchParams = defineParameters(
    EnumParam("algorithm", "Algorithm", 2, "Cepstrum",
              "Harmonic Product Spectrum", "YinFFT"),
    FloatParam("minFreq", "Low Frequency Bound", 20, Min(0), Max(10000),
               UpperLimit<kMaxFreq>()),
    FloatParam("maxFreq", "High Frequency Bound", 10000, Min(1), Max(20000),
               LowerLimit<kMinFreq>()),
    EnumParam("unit", "Frequency Unit", 0, "Hz", "MIDI"),
    FFTParam<kMaxFFTSize>("fftSettings", "FFT Settings", 1024, -1, -1),
    LongParam<Fixed<true>>("maxFFTSize", "Maxiumm FFT Size", 16384, Min(4),
                           PowerOfTwo{}));

class PitchClient : public FluidBaseClient, public AudioIn, public ControlOut
{
  using size_t = std::size_t;
  using CepstrumF0 = algorithm::CepstrumF0;
  using HPS = algorithm::HPS;
  using YINFFT = algorithm::YINFFT;

public:
  using ParamDescType = decltype(PitchParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors() { return PitchParams; }

  PitchClient(ParamSetViewType& p)
      : mParams(p), mSTFTBufferedProcess(get<kMaxFFTSize>(), 1, 0),
        cepstrumF0(get<kMaxFFTSize>())
  {
    audioChannelsIn(1);
    controlChannelsOut({1,2});
    setInputLabels({"audio input"});
    setOutputLabels({"pitch (hz or MIDI), pitch confidence (0-1)"});
    mDescriptors = FluidTensor<double, 1>(2);
  }

  template <typename T>
  void process(std::vector<HostVector<T>>& input,
               std::vector<HostVector<T>>& output, FluidContext& c)
  {
    if (!input[0].data() || !output[0].data()) return;
    assert(FluidBaseClient::controlChannelsOut().size && "No control channels");
    assert(asSigned(output.size()) >= FluidBaseClient::controlChannelsOut().size &&
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
    // pitch
    if(get<kUnit>() == 1){
      output[0](0) = mDescriptors(0) == 0? -999:
      69 + (12 * log2(mDescriptors(0) / 440.0));
    }
    else {
      output[0](0) = mDescriptors(0);
    }
    // pitch confidence
    output[1](0) = static_cast<T>(mDescriptors(1));
  }
  index latency() { return get<kFFT>().winSize(); }
  index controlRate() { return get<kFFT>().hopSize(); }
  void  reset()
  {
    mSTFTBufferedProcess.reset();
    cepstrumF0.init(get<kFFT>().frameSize());
    mMagnitude.resize(get<kFFT>().frameSize());
  }

private:
  ParameterTrackChanges<index, double>        mParamTracker;
  STFTBufferedProcess<ParamSetViewType, kFFT> mSTFTBufferedProcess;

  CepstrumF0             cepstrumF0;
  HPS                    hps;
  YINFFT                 yinFFT;
  FluidTensor<double, 1> mMagnitude;
  FluidTensor<double, 1> mDescriptors;
};
} // namespace pitch

using RTPitchClient = ClientWrapper<pitch::PitchClient>;

auto constexpr NRTPitchParams = makeNRTParams<pitch::PitchClient>(
    InputBufferParam("source", "Source Buffer"),
    BufferParam("features", "Features Buffer"));

using NRTPitchClient =
    NRTControlAdaptor<pitch::PitchClient, decltype(NRTPitchParams),
                      NRTPitchParams, 1, 1>;

using NRTThreadedPitchClient = NRTThreadingAdaptor<NRTPitchClient>;

} // namespace client
} // namespace fluid
