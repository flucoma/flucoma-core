/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright University of Huddersfield.
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

enum OnsetParamIndex { kFunction, kFilterSize, kFrameDelta, kFFT };

constexpr auto OnsetFeatureParams =
    defineParameters(EnumParam("metric", "Spectral Change Metric", 0, "Energy",
                         "High Frequency Content", "Spectral Flux",
                         "Modified Kullback-Leibler", "Itakura-Saito", "Cosine",
                         "Phase Deviation", "Weighted Phase Deviation",
                         "Complex Domain", "Rectified Complex Domain"),
        LongParam("filterSize", "Filter Size", 5, Min(1), Odd(), Max(101)),
        LongParam("frameDelta", "Frame Delta", 0, Min(0), Max(8192)),
        FFTParam("fftSettings", "FFT Settings", 1024, -1, -1));

class OnsetFeatureClient : public FluidBaseClient,
                           public AudioIn,
                           public ControlOut
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

  static constexpr auto& getParameterDescriptors()
  {
    return OnsetFeatureParams;
  }

  OnsetFeatureClient(ParamSetViewType& p, FluidContext& c)
      : mParams{p},
        mAlgorithm{get<kFFT>().max(), 101, c.allocator()},
        mBufferedProcess{get<kFFT>().max() + 8192, 0, 1, 0, c.hostVectorSize(),
            c.allocator()}
  {
    audioChannelsIn(1);
    controlChannelsOut({1, 1});
    setInputLabels({"audio input"});
    setOutputLabels({"spectral difference"});
  }

  template <typename T>
  void process(std::vector<HostVector<T>>& input,
      std::vector<HostVector<T>>& output, FluidContext& c)
  {
    if (!input[0].data() || !output[0].data()) return;

    index totalWindow = get<kFFT>().winSize();
    if (get<kFunction>() > 1 && get<kFunction>() < 5)
      totalWindow += get<kFrameDelta>();

    if (mParamsTracker.changed(get<kFFT>().fftSize(), get<kFFT>().winSize()))
    {
      mAlgorithm.init(
          get<kFFT>().winSize(), get<kFFT>().fftSize(), get<kFilterSize>());
    }
    
    if (mHostSizeTracker.changed(c.hostVectorSize()))
    {
      mBufferedProcess = BufferedProcess{get<kFFT>().max() + 8192, 0, 1, 0, c.hostVectorSize(),
            c.allocator()};
    }

    mBufferedProcess.push(FluidTensorView<T, 2>(input[0]));
    mBufferedProcess.processInput(
        totalWindow, get<kFFT>().hopSize(), c, [&](RealMatrixView in) {
          mDescriptor = mAlgorithm.processFrame(in.row(0), get<kFunction>(),
              get<kFilterSize>(), get<kFrameDelta>(), c.allocator());
        });

    output[0](0) = static_cast<T>(mDescriptor);
  }

  index latency() const { return static_cast<index>(get<kFFT>().hopSize()); }

  AnalysisSize analysisSettings()
  {
    return {get<kFFT>().winSize(), get<kFFT>().hopSize()};
  }

  void reset(FluidContext&)
  {
    mBufferedProcess.reset();
    mAlgorithm.init(
        get<kFFT>().winSize(), get<kFFT>().fftSize(), get<kFilterSize>());
  }

private:
  OnsetDetectionFunctions                    mAlgorithm;
  double                                     mDescriptor;
  ParameterTrackChanges<index, index>        mParamsTracker;
  ParameterTrackChanges<index>               mHostSizeTracker;
  BufferedProcess                            mBufferedProcess;
};
} // namespace onsetfeature

using RTOnsetFeatureClient = ClientWrapper<onsetfeature::OnsetFeatureClient>;

auto constexpr NRTOnsetFeatureParams =
    makeNRTParams<onsetfeature::OnsetFeatureClient>(
        InputBufferParam("source", "Source Buffer"),
        BufferParam("features", "Feature Buffer"));


using NRTOnsetFeatureClient =
    NRTControlAdaptor<onsetfeature::OnsetFeatureClient,
        decltype(NRTOnsetFeatureParams), NRTOnsetFeatureParams, 1, 1>;


using NRTThreadedOnsetFeatureClient =
    NRTThreadingAdaptor<NRTOnsetFeatureClient>;

} // namespace client
} // namespace fluid
