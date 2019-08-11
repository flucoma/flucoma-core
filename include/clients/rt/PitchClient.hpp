#pragma once

#include "BufferedProcess.hpp"
#include "../common/AudioClient.hpp"
#include "../common/FluidBaseClient.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/ParameterTypes.hpp"
#include "../nrt/FluidNRTClientWrapper.hpp"
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

auto constexpr PitchParams = defineParameters(
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


class PitchClient : public FluidBaseClient<decltype(PitchParams), PitchParams>,
                    public AudioIn,
                    public ControlOut {
  using size_t = std::size_t;
  using CepstrumF0 = algorithm::CepstrumF0;
  using HPS = algorithm::HPS;
  using YINFFT = algorithm::YINFFT;

public:
  PitchClient(ParamSetViewType &p)
      : FluidBaseClient(p), mSTFTBufferedProcess(get<kMaxFFTSize>(), 1, 0) {
    FluidBaseClient::audioChannelsIn(1);
    FluidBaseClient::controlChannelsOut(2);
    mDescriptors = FluidTensor<double, 1>(2);
  }

  template <typename T>
  void process(std::vector<HostVector<T>> &input, std::vector<HostVector<T>> &output, FluidContext& c,
               bool reset = false) {
    if (!input[0].data() || !output[0].data())
      return;
    assert(FluidBaseClient::controlChannelsOut() && "No control channels");
    assert(output.size() >= FluidBaseClient::controlChannelsOut() &&
           "Too few output channels");

    if (mParamTracker.changed(get<kFFT>().frameSize())) {
      cepstrumF0.init(get<kFFT>().frameSize());
      mMagnitude.resize(get<kFFT>().frameSize());
    }

    mSTFTBufferedProcess.processInput(
        mParams, input, c, reset, [&](ComplexMatrixView in) {
          algorithm::STFT::magnitude(in.row(0), mMagnitude);
          switch (get<kAlgorithm>()) {
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
    output[0](0) = get<kUnit>() == 0
                       ? mDescriptors(0)
                       : 69 + (12 * log2(mDescriptors(0) / 440.0)); // pitch
    output[1](0) = mDescriptors(1); // pitch confidence
  }
  size_t latency() { return get<kFFT>().winSize(); }
  size_t controlRate() { return get<kFFT>().hopSize(); }

private:
  ParameterTrackChanges<size_t> mParamTracker;
  STFTBufferedProcess<ParamSetViewType, kFFT> mSTFTBufferedProcess;
  CepstrumF0 cepstrumF0;
  HPS hps;
  YINFFT yinFFT;
  FluidTensor<double, 1> mMagnitude;
  FluidTensor<double, 1> mDescriptors;
};

auto constexpr NRTPitchParams =
    makeNRTParams<PitchClient>({InputBufferParam("source", "Source Buffer")},
                               {BufferParam("features", "Features Buffer")});

using NRTPitchClient =
    NRTControlAdaptor<PitchClient, decltype(NRTPitchParams), NRTPitchParams,
                      1, 1>;

using NRTThreadedPitchClient = NRTThreadingAdaptor<NRTPitchClient>;

} // namespace client
} // namespace fluid
