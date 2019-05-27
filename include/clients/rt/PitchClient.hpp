#pragma once

#include "../../algorithms/public/CepstrumF0.hpp"
#include "../../algorithms/public/HPS.hpp"
#include "../../algorithms/public/YINFFT.hpp"
#include "../../data/TensorTypes.hpp"
#include "../common/AudioClient.hpp"
#include "../common/FluidBaseClient.hpp"
#include "../common/FluidContext.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/ParameterTypes.hpp"
#include "../nrt/FluidNRTClientWrapper.hpp"
#include "../rt/BufferedProcess.hpp"
#include <tuple>

namespace fluid {
namespace client {

enum PitchParamIndex { kAlgorithm, kFFT, kMaxFFTSize };

auto constexpr PitchParams = defineParameters(
    EnumParam("algorithm", "Algorithm", 2, "Cepstrum", "Harmonic Product Spectrum", "YinFFT"),
    FFTParam<kMaxFFTSize>("fftSettings", "FFT Settings", 1024, -1, -1),
    LongParam<Fixed<true>>("maxFFTSize", "Maxiumm FFT Size", 16384, Min(4),
                           PowerOfTwo{}));

template <typename T>
class PitchClient : public FluidBaseClient<decltype(PitchParams), PitchParams>,
                    public AudioIn,
                    public ControlOut {
  using HostVector = HostVector<T>;
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

  void process(std::vector<HostVector> &input,
               std::vector<HostVector> &output, FluidContext& c) {
    if (!input[0].data() || !output[0].data())
      return;
    assert(FluidBaseClient::controlChannelsOut() && "No control channels");
    assert(output.size() >= FluidBaseClient::controlChannelsOut() &&
           "Too few output channels");

    if (mWinSizeTracker.changed(get<kFFT>().frameSize())) {
      mMagnitude.resize(get<kFFT>().frameSize());
    }

    mSTFTBufferedProcess.processInput(
        mParams, input, c, [&](ComplexMatrixView in) {
          algorithm::STFT::magnitude(in.row(0), mMagnitude);
          switch (get<kAlgorithm>()) {
          case 0:
            cepstrumF0.processFrame(mMagnitude, mDescriptors, sampleRate());
            break;
          case 1:
            hps.processFrame(mMagnitude, mDescriptors, 4, sampleRate());
            break;
          case 2:
            yinFFT.processFrame(mMagnitude, mDescriptors, sampleRate());
            break;
          }
        });
    output[0](0) = mDescriptors(0); // pitch
    output[1](0) = mDescriptors(1); // pitch confidence
  }
  size_t latency() { return get<kFFT>().winSize(); }
  size_t controlRate() { return get<kFFT>().hopSize(); }

private:
  ParameterTrackChanges<size_t> mWinSizeTracker;
  STFTBufferedProcess<ParamSetViewType, T, kFFT> mSTFTBufferedProcess;
  CepstrumF0 cepstrumF0;
  HPS hps;
  YINFFT yinFFT;
  FluidTensor<double, 1> mMagnitude;
  FluidTensor<double, 1> mDescriptors;
};

auto constexpr NRTPitchParams =
    makeNRTParams<PitchClient>({BufferParam("source", "Source Buffer")},
                               {BufferParam("features", "Features Buffer")});

template <typename T>
using NRTPitchClient =
    NRTControlAdaptor<PitchClient<T>, decltype(NRTPitchParams), NRTPitchParams,
                      1, 1>;

template <typename T>
using NRTThreadedPitchClient = NRTThreadingAdaptor<NRTPitchClient<T>>;
    
} // namespace client
} // namespace fluid
