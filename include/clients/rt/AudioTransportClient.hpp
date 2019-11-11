#pragma once

#include "../../algorithms/AudioTransport.hpp"
#include "clients/common/FluidBaseClient.hpp"
#include "clients/common/ParameterConstraints.hpp"
#include "clients/common/ParameterSet.hpp"
#include "clients/common/ParameterTrackChanges.hpp"
#include "clients/common/ParameterTypes.hpp"
#include "clients/nrt/FluidNRTClientWrapper.hpp"
#include "clients/rt/BufferedProcess.hpp"
#include <clients/common/AudioClient.hpp>
#include <tuple>

namespace fluid {
namespace client {

extern auto constexpr AudioTransportParams = defineParameters();

class AudioTransportClient : public FluidBaseClient,
                             public AudioIn,
                             public AudioOut {



  enum AudioTransportParamTags {
    kInterpolation,
    kBandwidth,
    kFFT,
    kMaxFFTSize
  };

public:
  FLUID_DECLARE_PARAMS(FloatParam("interpolation", "Interpolation", 0.0,
                                  Min(0.0), Max(1.0)),
                       LongParam("bandwidth", "Bandwidth", 255, Min(255),
                                 FrameSizeUpperLimit<kFFT>()),
                       FFTParam<kMaxFFTSize>("fftSettings", "FFT Settings",
                                             1024, -1, -1,
                                             FrameSizeLowerLimit<kBandwidth>()),
                       LongParam<Fixed<true>>("maxFFTSize", "Maxiumm FFT Size",
                                              16384, Min(4), PowerOfTwo{}));
  AudioTransportClient(ParamSetViewType &p)
      : mSTFTBufferedProcess{static_cast<size_t>(get<kMaxFFTSize>()), 2, 1}, mParams(p), mAlgorithm(get<kMaxFFTSize>()) {
    audioChannelsIn(2);
    audioChannelsOut(1);
  }

  template <typename T>
  void process(std::vector<FluidTensorView<T, 1>> &input,
               std::vector<FluidTensorView<T, 1>> &output, FluidContext &c,
               bool reset = false) {
    if (!input[0].data() || !input[1].data())
      return;
    if (!mAlgorithm.initialized() ||
        mTracking.changed(get<kFFT>().winSize(), get<kFFT>().hopSize(),
                             get<kFFT>().fftSize(), get<kBandwidth>())) {
      mAlgorithm.init(get<kFFT>().winSize(), get<kFFT>().fftSize(),
                      get<kFFT>().hopSize(), get<kBandwidth>());
    }
    mSTFTBufferedProcess.process(
        mParams, input, output, c, reset,
        [this](ComplexMatrixView in, ComplexMatrixView out) {
      mAlgorithm.processFrame(in.row(0), in.row(1), get<kInterpolation>(), out.row(0));
        });
  }

  size_t latency() { return get<kFFT>().winSize(); }

private:
  STFTBufferedProcess<ParamSetViewType, kFFT> mSTFTBufferedProcess;
  algorithm::AudioTransport mAlgorithm;
  ParameterTrackChanges<size_t, size_t, size_t, size_t> mTracking;
};

using RTAudioTransportClient = ClientWrapper<AudioTransportClient>;
auto constexpr NRTAudioTransportParams = makeNRTParams<AudioTransportClient>(
    {InputBufferParam("source1", "Source Buffer 1"),
     InputBufferParam("source2", "Source Buffer 2")},
    {BufferParam("out", "output Buffer")});

using NRTAudioTransport =
    NRTStreamAdaptor<RTAudioTransportClient, decltype(NRTAudioTransportParams),
                     NRTAudioTransportParams, 2, 1>;

using NRTThreadedAudioTransportClient = NRTThreadingAdaptor<NRTAudioTransport>;

} // namespace client
} // namespace fluid
