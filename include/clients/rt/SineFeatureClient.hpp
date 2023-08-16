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
#include "../common/ParameterTrackChanges.hpp"
#include "../common/ParameterTypes.hpp"
#include "../../algorithms/public/SineFeature.hpp"
#include <tuple>

namespace fluid {
namespace client {
namespace sinefeature {

enum SineFeatureParamIndex {
  kNPeaks,
  kDetectionThreshold,
  kSortBy,
  kLogFreq,
  kLogMag,
  kFFT
};

constexpr auto SineFeatureParams = defineParameters(
    LongParamRuntimeMax<Primary>("numPeaks", "Number of Sinusoidal Peaks", 10,
                                 Min(1), FrameSizeUpperLimit<kFFT>()),
    FloatParam("detectionThreshold", "Peak Detection Threshold", -96, Min(-144),
               Max(0)),
    EnumParam("order", "Sort Peaks Output", 0, "Frequencies", "Amplitudes"),
    EnumParam("freqUnit", "Units for Frequencies", 0, "Hz", "MIDI"),
    EnumParam("magUnit", "Units for Magnitudes", 0, "Amp", "dB"),
    FFTParam("fftSettings", "FFT Settings", 1024, -1, -1));

class SineFeatureClient : public FluidBaseClient,
                          public AudioIn,
                          public ControlOut
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
    controlChannelsOut({2, get<kNPeaks>(), get<kNPeaks>().max()});
  }

  static constexpr auto& getParameterDescriptors() { return SineFeatureParams; }

  SineFeatureClient(ParamSetViewType& p, FluidContext& c)
      : mParams(p), mSTFTBufferedProcess{get<kFFT>(), 1, 2, c.hostVectorSize(),
                                         c.allocator()},
        mSineFeature{c.allocator()},
        mPeaks(get<kNPeaks>().max(), c.allocator()),
        mMags(get<kNPeaks>().max(), c.allocator())
  {
    audioChannelsIn(1);
    controlChannelsOut({2, get<kNPeaks>(), get<kNPeaks>().max()});
    setInputLabels({"Audio Input"});
    setOutputLabels({"Peak Frequencies", "Peak Magnitudes"});
  }

  template <typename T>
  void process(std::vector<HostVector<T>>& input,
               std::vector<HostVector<T>>& output, FluidContext& c)
  {
    if (!input[0].data() || !output[0].data()) return;
    
    index nPeaks = get<kNPeaks>();

    if (!mSineFeature.initialized() ||
        mTrackValues.changed(get<kFFT>().winSize(), get<kFFT>().fftSize(),
                             nPeaks, sampleRate()))
    {
      mSineFeature.init(get<kFFT>().winSize(), get<kFFT>().fftSize());
      controlChannelsOut({2, nPeaks});
    }

    if (mHostSizeTracker.changed(c.hostVectorSize()))
    {
      mSTFTBufferedProcess = STFTBufferedProcess<false>(
          get<kFFT>(), 1, 0, c.hostVectorSize(), c.allocator());
    }

    auto peaks = mPeaks(Slice(0, nPeaks));
    auto mags = mMags(Slice(0, nPeaks));

    mSTFTBufferedProcess.processInput(
        get<kFFT>(), input, c, [&](ComplexMatrixView in) {
          mNumPeaks = mSineFeature.processFrame(
              in.row(0), peaks, mags, sampleRate(), get<kDetectionThreshold>(),
              get<kSortBy>(), c.allocator());
        });

    auto validPeaks = mPeaks(Slice(0, mNumPeaks));
    auto validMags = mMags(Slice(0, mNumPeaks));

    if (get<kLogFreq>())
    {
      std::transform(validPeaks.begin(), validPeaks.end(), output[0].begin(),
                     [](auto peak) {
                       constexpr auto ratio = 1 / 440.0;
                       return peak == 0 ? -999
                                        : 69 + (12 * std::log2(peak * ratio));
                     });
      output[0](Slice(mNumPeaks, get<kNPeaks>().max() - mNumPeaks)).fill(-999);
    }
    else
    {
      output[0](Slice(0, mNumPeaks)) <<= validPeaks;
      output[0](Slice(mNumPeaks, get<kNPeaks>().max() - mNumPeaks)).fill(0);
    }
    
    if (get<kLogMag>())
    {
      output[1](Slice(0, mNumPeaks)) <<= validMags;
      output[1](Slice(mNumPeaks, get<kNPeaks>().max() - mNumPeaks)).fill(-144);
    }
    else
    {
      std::transform(validMags.begin(), validMags.end(), output[1].begin(),
                     [](auto peak) { return std::pow(10, (peak / 20)); });
      output[1](Slice(mNumPeaks, get<kNPeaks>().max() - mNumPeaks)).fill(0);
    }
  }

  index latency() const { return get<kFFT>().winSize(); }
  
  void reset(FluidContext&)
  {
    mSTFTBufferedProcess.reset();
    mSineFeature.init(get<kFFT>().winSize(), get<kFFT>().fftSize());
  }

  AnalysisSize analysisSettings()
  {
    return {get<kFFT>().winSize(), get<kFFT>().hopSize()};
  }

private:
  STFTBufferedProcess<false>                  mSTFTBufferedProcess;
  algorithm::SineFeature                      mSineFeature;
  ParameterTrackChanges<index, index, index, double> mTrackValues;
  ParameterTrackChanges<index>                mHostSizeTracker;
  FluidTensor<double, 1>                      mPeaks;
  FluidTensor<double, 1>                      mMags;
  index                                       mNumPeaks;
};

} // namespace sinefeature
using RTSineFeatureClient = ClientWrapper<sinefeature::SineFeatureClient>;

auto constexpr NRTSineFeatureParams =
    makeNRTParams<sinefeature::SineFeatureClient>(
        InputBufferParam("source", "Source Buffer"),
        BufferParam("frequency", "Peak Frequencies Buffer"),
        BufferParam("magnitude", "Peak Magnitudes Buffer"));

using NRTSineFeatureClient = NRTControlAdaptor<sinefeature::SineFeatureClient,
                                               decltype(NRTSineFeatureParams),
                                               NRTSineFeatureParams, 1, 2>;

using NRTThreadedSineFeatureClient = NRTThreadingAdaptor<NRTSineFeatureClient>;

} // namespace client
} // namespace fluid
