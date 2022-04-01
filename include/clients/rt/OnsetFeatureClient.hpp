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
#include "../../algorithms/public/OnsetDetectionFunctions.hpp"
#include "../../data/TensorTypes.hpp"
#include <tuple>

namespace fluid {
namespace client {
namespace onsetfeature {

enum OnsetParamIndex {
  kFunction,
  kFilterSize,
  kFrameDelta,
  kFFT,
  kMaxFFTSize
};

constexpr auto OnsetFeatureParams = defineParameters(
    EnumParam("metric", "Spectral Change Metric", 0, "Energy",
              "High Frequency Content", "Spectral Flux",
              "Modified Kullback-Leibler", "Itakura-Saito", "Cosine",
              "Phase Deviation", "Weighted Phase Deviation", "Complex Domain",
              "Rectified Complex Domain"),
    LongParam("filterSize", "Filter Size", 5, Min(1), Odd(), Max(101)),
    LongParam("frameDelta", "Frame Delta", 0, Min(0)),
    FFTParam<kMaxFFTSize>("fftSettings", "FFT Settings", 1024, -1, -1),
    LongParam<Fixed<true>>("maxFFTSize", "Maxiumm FFT Size", 16384, Min(4),
                           PowerOfTwo{}));

class OnsetFeatureClient : public FluidBaseClient, public AudioIn, public ControlOut
{

  using OnsetDetectionFunctions = algorithm::OnsetDetectionFunctions;

public:
  using ParamDescType = decltype(OnsetFeatureParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors() { return OnsetFeatureParams; }

  OnsetFeatureClient(ParamSetViewType& p)
      : mParams{p}, mAlgorithm{get<kMaxFFTSize>()}
  {
    audioChannelsIn(1);
    controlChannelsOut({1,1});
    setInputLabels({"audio input"});
    setOutputLabels({"1 when slice detected, 0 otherwise"});
  }

  template <typename T>
  void process(std::vector<HostVector<T>>& input,
               std::vector<HostVector<T>>& output, FluidContext& c)
  {
    using std::size_t;

    if (!input[0].data() || !output[0].data()) return;

    index hostVecSize = input[0].size();
    index totalWindow = get<kFFT>().winSize();
    if (get<kFunction>() > 1 && get<kFunction>() < 5)
      totalWindow += get<kFrameDelta>();
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
    
    mBufferedProcess.push(RealMatrixView(in));
    mBufferedProcess.processInput(
        totalWindow, get<kFFT>().hopSize(), c, [&, this](RealMatrixView in) {
          mDescriptor = mAlgorithm.processFrame(
              in.row(0), get<kFunction>(), get<kFilterSize>(), get<kFrameDelta>());
        });

    output[0](0) = static_cast<T>(mDescriptor);
  }

  index latency() { return static_cast<index>(get<kFFT>().hopSize()); }

  AnalysisSize analysisSettings()
  {
    return {get<kFFT>().winSize(), get<kFFT>().hopSize()};
  }

  void reset()
  {    
    mBufferedProcess.reset();
    mAlgorithm.init(get<kFFT>().winSize(), get<kFFT>().fftSize(),
                    get<kFilterSize>());
  }

private:
  OnsetDetectionFunctions                    mAlgorithm;
  double                                     mDescriptor;
  ParameterTrackChanges<index, index, index> mBufferParamsTracker;
  ParameterTrackChanges<index, index>        mParamsTracker;
  BufferedProcess                            mBufferedProcess;
};
} // namespace onsetfeature

using RTOnsetFeatureClient = ClientWrapper<onsetfeature::OnsetFeatureClient>;

auto constexpr NRTOnsetFeatureParams =
    makeNRTParams<onsetfeature::OnsetFeatureClient>(
        InputBufferParam("source", "Source Buffer"),
        BufferParam("features", "Feature Buffer"));


using NRTOnsetFeatureClient =
    NRTControlAdaptor<onsetfeature::OnsetFeatureClient, decltype(NRTOnsetFeatureParams),
                    NRTOnsetFeatureParams, 1, 1>;


using NRTThreadedOnsetFeatureClient = NRTThreadingAdaptor<NRTOnsetFeatureClient>;

} // namespace client
} // namespace fluid
