#pragma once

#include "../common/AudioClient.hpp"
#include "../common/BufferedProcess.hpp"
#include "../common/FluidBaseClient.hpp"
#include "../common/FluidNRTClientWrapper.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/ParameterTrackChanges.hpp"
#include "../common/ParameterTypes.hpp"
#include "../../algorithms/public/AudioTransport.hpp"
#include <tuple>

namespace fluid {
namespace client {
namespace audiotransport {

enum AudioTransportParamTags { kInterpolation, kFFT, kMaxFFTSize };

constexpr auto AudioTransportParams = defineParameters(
    FloatParam("interpolation", "Interpolation", 0.0, Min(0.0), Max(1.0)),
    FFTParam<kMaxFFTSize>("fftSettings", "FFT Settings", 1024, -1, -1),
    LongParam<Fixed<true>>("maxFFTSize", "Maxiumm FFT Size", 16384, Min(4),
                           PowerOfTwo{}));

class AudioTransportClient : public FluidBaseClient,
                             public AudioIn,
                             public AudioOut
{
public:
  using ParamDescType = decltype(AudioTransportParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors()
  {
    return AudioTransportParams;
  }

  AudioTransportClient(ParamSetViewType& p)
      : mParams{p}, mAlgorithm(get<kMaxFFTSize>())
  {
    audioChannelsIn(2);
    audioChannelsOut(1);
  }

  template <typename T>
  void process(std::vector<FluidTensorView<T, 1>>& input,
               std::vector<FluidTensorView<T, 1>>& output, FluidContext& c)
  {
    if (!input[0].data() || !input[1].data()) return;
    index hostVecSize = input[0].size();

    if (!mAlgorithm.initialized() ||
        mTracking.changed(get<kFFT>().winSize(), get<kFFT>().hopSize(),
                          get<kFFT>().fftSize()))
    {
      mAlgorithm.init(get<kFFT>().winSize(), get<kFFT>().fftSize(),
                      get<kFFT>().hopSize());
      mBufferedProcess.hostSize(hostVecSize);
      mBufferedProcess.maxSize(get<kFFT>().winSize(), get<kFFT>().winSize(), 2,
                               2);
    }
    RealMatrix in(2, input[0].size());
    in.row(0) = input[0];
    in.row(1) = input[1];
    mBufferedProcess.push(RealMatrixView(in));
    mBufferedProcess.process(
        get<kFFT>().winSize(), get<kFFT>().winSize(), get<kFFT>().hopSize(), c,
        [&](RealMatrixView _in, RealMatrixView _out) {
          mAlgorithm.processFrame(_in.row(0), _in.row(1), get<kInterpolation>(),
                                  _out);
        });
    RealMatrix out(2, hostVecSize);
    mBufferedProcess.pull(RealMatrixView(out));
    RealVectorView result = out.row(0);
    RealVectorView norm = out.row(1);
    for (index i = 0; i < result.size(); i++)
    { result(i) /= (norm(i) > 0 ? norm(i) : 1); }
    if (output[0].data()) output[0] = result;
  }

  index latency() { return get<kFFT>().winSize(); }
  void  reset() { mBufferedProcess.reset(); }

private:
  BufferedProcess                            mBufferedProcess;
  algorithm::AudioTransport                  mAlgorithm;
  ParameterTrackChanges<index, index, index> mTracking;
};
} // namespace audiotransport

using RTAudioTransportClient =
    ClientWrapper<audiotransport::AudioTransportClient>;

auto constexpr NRTAudioTransportParams =
    makeNRTParams<audiotransport::AudioTransportClient>(
        InputBufferParam("source1", "Source Buffer 1"),
        InputBufferParam("source2", "Source Buffer 2"),
        BufferParam("destination", "Destination Buffer"));

using NRTAudioTransport = NRTStreamAdaptor<audiotransport::AudioTransportClient,
                                           decltype(NRTAudioTransportParams),
                                           NRTAudioTransportParams, 2, 1>;

using NRTThreadedAudioTransportClient = NRTThreadingAdaptor<NRTAudioTransport>;

} // namespace client
} // namespace fluid
