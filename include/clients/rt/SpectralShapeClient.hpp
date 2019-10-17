#pragma once

#include "BufferedProcess.hpp"
#include "../common/AudioClient.hpp"
#include "../common/FluidBaseClient.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/ParameterTypes.hpp"
#include "../common/FluidNRTClientWrapper.hpp"
#include "../../algorithms/public/SpectralShape.hpp"
#include "../../data/TensorTypes.hpp"

#include <tuple>

namespace fluid {
namespace client {

using algorithm::SpectralShape;

enum SpectralShapeParamIndex { kFFT, kMaxFFTSize };

auto constexpr SpectralShapeParams = defineParameters(
    FFTParam<kMaxFFTSize>("fftSettings", "FFT Settings", 1024, -1, -1),
    LongParam<Fixed<true>>("maxFFTSize", "Maxiumm FFT Size", 16384, Min(4),
                           PowerOfTwo{}));

template <typename T>
class SpectralShapeClient
    : public FluidBaseClient<decltype(SpectralShapeParams),
                             SpectralShapeParams>,
      public AudioIn,
      public ControlOut {
  using HostVector = FluidTensorView<T,1>;

public:
  SpectralShapeClient(ParamSetViewType &p)
      : FluidBaseClient(p), mSTFTBufferedProcess(get<kMaxFFTSize>(), 1, 0) {
    FluidBaseClient::audioChannelsIn(1);
    FluidBaseClient::controlChannelsOut(7);
    mDescriptors = FluidTensor<double, 1>(7);
  }

  void process(std::vector<HostVector> &input,
               std::vector<HostVector> &output, FluidContext& c, bool reset = false) {
    using std::size_t;

    if (!input[0].data() || !output[0].data())
      return;
    assert(FluidBaseClient::controlChannelsOut() && "No control channels");
    assert(output.size() >= FluidBaseClient::controlChannelsOut() &&
           "Too few output channels");

    if (mWinSizeTracker.changed(get<kFFT>().frameSize())) {
      mMagnitude.resize(get<kFFT>().frameSize());
      mBinHz = sampleRate() / get<kFFT>().fftSize();
    }

    mSTFTBufferedProcess.processInput(
        mParams, input, c, reset, [&](ComplexMatrixView in) {
          algorithm::STFT::magnitude(in.row(0), mMagnitude);
          mAlgorithm.processFrame(mMagnitude, mDescriptors);
        });

    for (int i = 0; i < 7; ++i){
      //TODO: probably move this logic to algorithm
      if(i==0||i==1||i==4)output[i](0) =  mBinHz * mDescriptors(i);
      else output[i](0) = mDescriptors(i);
    }

  }

  size_t latency() { return get<kFFT>().winSize(); }

  size_t controlRate() { return get<kFFT>().hopSize(); }

private:
  ParameterTrackChanges<size_t> mWinSizeTracker;
  STFTBufferedProcess<ParamSetViewType, T, kFFT> mSTFTBufferedProcess;
  SpectralShape mAlgorithm{get<kMaxFFTSize>()};
  FluidTensor<double, 1> mMagnitude;
  FluidTensor<double, 1> mDescriptors;
  double mBinHz;
};

auto constexpr NRTSpectralShapeParams = makeNRTParams<SpectralShapeClient>(
    {InputBufferParam("source", "Source Buffer")},
    {BufferParam("features", "Features Buffer")});

template <typename T>
using NRTSpectralShapeClient =
    NRTControlAdaptor<SpectralShapeClient<T>, decltype(NRTSpectralShapeParams),
                      NRTSpectralShapeParams, 1, 1>;

template <typename T>
using NRTThreadedSpectralShapeClient = NRTThreadingAdaptor<NRTSpectralShapeClient<T>>;

} // namespace client
} // namespace fluid
