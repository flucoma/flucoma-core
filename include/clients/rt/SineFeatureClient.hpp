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
#include "../common/ParameterTrackChanges.hpp"
#include "../common/ParameterTypes.hpp"
#include "../../algorithms/public/SineFeatureExtraction.hpp"
#include <tuple>

namespace fluid {
namespace client {
namespace sinefeature {

enum SineFeatureParamIndex {
  kNPeaks,
  kDetectionThreshold,
  kSortBy,
  kFFT
};

constexpr auto SineFeatureParams = defineParameters(
    LongParamRuntimeMax<Primary>("numPeaks", "Number of Sinusoidal Peaks", 10,
                        Min(1),
                        FrameSizeUpperLimit<kFFT>()),
    FloatParam("detectionThreshold", "Peak Detection Threshold", -96, Min(-144),
               Max(0)),
    EnumParam("sortBy", "Sort Peaks Output", 0, "Nothing", "Frequencies", "Amplitudes"),
    FFTParam("fftSettings", "FFT Settings", 1024, -1, -1));

class SineFeatureClient : public FluidBaseClient, public AudioIn, public ControlOut
{

public:
  using ParamDescType = decltype(SineFeatureParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  template <size_t N> 
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  void setParams(ParamSetViewType& p) 
  { 
    mParams = p; 
    controlChannelsOut({1, get<kNPeaks>(), get<kNPeaks>().max()});
  }

  static constexpr auto& getParameterDescriptors() { return SineFeatureParams; }

  SineFeatureClient(ParamSetViewType& p, FluidContext& c)
      : mParams(p), mSTFTBufferedProcess{get<kFFT>(), 1, 2, c.hostVectorSize(),
                                         c.allocator()},
        mSineFeatureExtractor{},
        mPeaks(get<kNPeaks>().max(), c.allocator())
  {
    audioChannelsIn(1);
    controlChannelsOut({1, get<kNPeaks>(), get<kNPeaks>().max()});
    setInputLabels({"Audio Input"});
    setOutputLabels({"Peak Frequencies"});
  }

  template <typename T>
  void process(std::vector<HostVector<T>>& input,
               std::vector<HostVector<T>>& output, FluidContext& c)
  {
    if (!input[0].data() || !output[0].data()) return;
    if (!mSineFeatureExtractor.initialized() ||
        mTrackValues.changed(get<kFFT>().winSize(), get<kFFT>().fftSize(),
                             sampleRate()))
    {
      mSineFeatureExtractor.init(get<kFFT>().winSize(), get<kFFT>().fftSize());
    }

    index nPeaks = get<kNPeaks>();
    auto peaks = mPeaks(Slice(0,nPeaks));

    mSTFTBufferedProcess.processInput(
        get<kFFT>(), input, c,
        [&](ComplexMatrixView in) { mSineFeatureExtractor.processFrame(
              in.row(0), peaks, sampleRate(),
              get<kDetectionThreshold>(), get<kSortBy>(),
              c.allocator());
        });
        
    // std::cout << mPeaks << "\n";
    
    output[0](Slice(0, nPeaks)) <<= peaks;
    output[0](Slice(nPeaks, get<kNPeaks>().max() - nPeaks)).fill(0);
  }

  index latency()
  {
    return get<kFFT>().winSize();
  }
  void reset()
  {
    mSTFTBufferedProcess.reset();
    mSineFeatureExtractor.init(get<kFFT>().winSize(), get<kFFT>().fftSize());
  }

private:
  STFTBufferedProcess<>                       mSTFTBufferedProcess;
  algorithm::SineFeatureExtraction            mSineFeatureExtractor;
  ParameterTrackChanges<index, index, double> mTrackValues;
  FluidTensor<double, 1>                      mPeaks;
};

} // namespace sinefeature
using RTSineFeatureClient = ClientWrapper<sinefeature::SineFeatureClient>;

auto constexpr NRTSineFeatureParams = makeNRTParams<sinefeature::SineFeatureClient>(
    InputBufferParam("source", "Source Buffer"),
    BufferParam("frequencies", "Peak Frequencies Buffer"),
    BufferParam("magnitudes", "Peak Magnitudes Buffer"));

using NRTSineFeatureClient =
    NRTStreamAdaptor<sinefeature::SineFeatureClient, decltype(NRTSineFeatureParams), NRTSineFeatureParams,
                     1, 2>;

using NRTThreadedSineFeatureClient = NRTThreadingAdaptor<NRTSineFeatureClient>;

} // namespace client
} // namespace fluid
