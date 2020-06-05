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
#include "../../algorithms/public/OnsetSegmentation.hpp"
#include "../../data/TensorTypes.hpp"
#include <tuple>

namespace fluid {
namespace client {

using algorithm::OnsetSegmentation;

enum OnsetParamIndex {
  kFunction,
  kThreshold,
  kDebounce,
  kFilterSize,
  kFrameDelta,
  kFFT,
  kMaxFFTSize
};

extern auto constexpr OnsetParams = defineParameters(
    EnumParam("metric", "Spectral Change Metric", 0, "Energy",
              "High Frequency Content", "Spectral Flux",
              "Modified Kullback-Leibler", "Itakura-Saito", "Cosine",
              "Phase Deviation", "Weighted Phase Deviation", "Complex Domain",
              "Rectified Complex Domain"),
    FloatParam("threshold", "Threshold", 0.5, Min(0)),
    LongParam("minSliceLength", "Minimum Length of Slice", 2, Min(0)),
    LongParam("filterSize", "Filter Size", 5, Min(1), Odd(), Max(101)),
    LongParam("frameDelta", "Frame Delta", 0, Min(0)),
    FFTParam<kMaxFFTSize>("fftSettings", "FFT Settings", 1024, -1, -1),
    LongParam<Fixed<true>>("maxFFTSize", "Maxiumm FFT Size", 16384, Min(4),
                           PowerOfTwo{}));

template <typename T>
class OnsetSliceClient
    : public FluidBaseClient<decltype(OnsetParams), OnsetParams>,
      public AudioIn,
      public AudioOut
{

  using HostVector = FluidTensorView<T, 1>;

public:
  OnsetSliceClient(ParamSetViewType& p) : FluidBaseClient(p)
  {
    FluidBaseClient::audioChannelsIn(1);
    FluidBaseClient::audioChannelsOut(1);
  }

  void process(std::vector<HostVector>& input, std::vector<HostVector>& output,
               FluidContext& c)
  {
    using algorithm::OnsetSegmentation;
    using std::size_t;

    if (!input[0].data() || !output[0].data()) return;

    index hostVecSize = input[0].size();
    index totalWindow = get<kFFT>().winSize() + get<kFrameDelta>();

    if (mBufferParamsTracker.changed(hostVecSize, get<kFFT>().winSize(),
                                     get<kFrameDelta>()))
    {
      mBufferedProcess.hostSize(hostVecSize);
      mBufferedProcess.maxSize(totalWindow, totalWindow,
                               FluidBaseClient::audioChannelsIn(),
                               FluidBaseClient::audioChannelsOut());
    }
    if (mParamsTracker.changed(get<kFFT>().fftSize(), get<kFFT>().winSize()))
    {
      mAlgorithm.init(get<kFFT>().winSize(), get<kFFT>().fftSize(),
                      get<kFilterSize>());
    }
    RealMatrix in(1, hostVecSize);
    in.row(0) = input[0];
    RealMatrix out(1, hostVecSize);
    int        frameOffset = 0; // in case kHopSize < hostVecSize
    mBufferedProcess.push(RealMatrixView(in));
    mBufferedProcess.processInput(
        totalWindow, get<kFFT>().hopSize(), c, [&, this](RealMatrixView in) {
          out.row(0)(frameOffset) = mAlgorithm.processFrame(
              in.row(0), get<kFunction>(), get<kFilterSize>(),
              get<kThreshold>(), get<kDebounce>(), get<kFrameDelta>());
          frameOffset += get<kFFT>().hopSize();
        });
    output[0] = out.row(0);
  }

  long latency() { return get<kFFT>().hopSize() + get<kFrameDelta>(); }
  void reset()
  {
    mBufferedProcess.reset();
    mAlgorithm.init(get<kFFT>().winSize(), get<kFFT>().fftSize(),
                    get<kFilterSize>());
  }

private:
  OnsetSegmentation                          mAlgorithm{get<kMaxFFTSize>()};
  ParameterTrackChanges<index, index, index> mBufferParamsTracker;
  ParameterTrackChanges<index, index>        mParamsTracker;
  BufferedProcess                            mBufferedProcess;
};

auto constexpr NRTOnsetSliceParams =
    makeNRTParams<OnsetSliceClient>(InputBufferParam("source", "Source Buffer"),
                                    BufferParam("indices", "Indices Buffer"));
template <typename T>
using NRTOnsetSliceClient =
    NRTSliceAdaptor<OnsetSliceClient<T>, decltype(NRTOnsetSliceParams),
                    NRTOnsetSliceParams, 1, 1>;

template <typename T>
using NRTThreadingOnsetSliceClient =
    NRTThreadingAdaptor<NRTOnsetSliceClient<T>>;

} // namespace client
} // namespace fluid
