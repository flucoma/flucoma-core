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
#include "../../algorithms/public/SineExtraction.hpp"
#include <tuple>

namespace fluid {
namespace client {
namespace sines {

enum SinesParamIndex {
  kBandwidth,
  kDetectionThreshold,
  kBirthLowThreshold,
  kBirthHighThreshold,
  kMinTrackLen,
  kTrackingMethod,
  kTrackMagRange,
  kTrackFreqRange,
  kTrackProb,
  kFFT
};

constexpr auto SineParams = defineParameters(
    LongParam("bandwidth", "Bandwidth", 76, Min(1),
              FrameSizeUpperLimit<kFFT>()),
    FloatParam("detectionThreshold", "Peak Detection Threshold", -96, Min(-144),
               Max(0)),
    FloatParam("birthLowThreshold", "Track Birth Low Frequency Threshold", -24,
               Min(-144), Max(0)),
    FloatParam("birthHighThreshold", "Track Birth High Frequency Threshold",
               -60, Min(-144), Max(0)),
    LongParam("minTrackLen", "Minimum Track Length", 15, Min(1)),
    EnumParam("trackingMethod", "Tracking Method", 0, "Greedy", "Hungarian"),
    FloatParam("trackMagRange", "Tracking Magnitude Range (dB)", 15., Min(1.),
               Max(200.)),
    FloatParam("trackFreqRange", "Tracking Frequency Range (Hz)", 50., Min(1.),
               Max(10000.)),
    FloatParam("trackProb", "Tracking Matching Probability", 0.5, Min(0.0),
               Max(1.0)),
    FFTParam("fftSettings", "FFT Settings", 1024, -1, -1,
                          FrameSizeLowerLimit<kBandwidth>()));

class SinesClient : public FluidBaseClient, public AudioIn, public AudioOut
{

public:
  using ParamDescType = decltype(SineParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors() { return SineParams; }

  SinesClient(ParamSetViewType& p)
      : mParams(p), mSTFTBufferedProcess{get<kFFT>().max(), 1, 2}
  {
    audioChannelsIn(1);
    audioChannelsOut(2);
    setInputLabels({"audio input"});
    setOutputLabels({"sinusoidal components","residual"});
  }

  template <typename T>
  void process(std::vector<HostVector<T>>& input,
               std::vector<HostVector<T>>& output, FluidContext& c)
  {
    if (!input[0].data()) return;
    if (!output[0].data() && !output[1].data()) return;
    if (!mSinesExtractor.initialized() ||
        mTrackValues.changed(get<kFFT>().winSize(), get<kFFT>().fftSize(),
                             sampleRate()))
    {
      mSinesExtractor.init(get<kFFT>().winSize(), get<kFFT>().fftSize(),
                           get<kFFT>().max());
    }

    mSTFTBufferedProcess.process(
        mParams, input, output, c,
        [this](ComplexMatrixView in, ComplexMatrixView out) {
          mSinesExtractor.processFrame(
              in.row(0), out.transpose(), sampleRate(),
              get<kDetectionThreshold>(), get<kMinTrackLen>(),
              get<kBirthLowThreshold>(), get<kBirthHighThreshold>(),
              get<kTrackingMethod>(), get<kTrackMagRange>(),
              get<kTrackFreqRange>(), get<kTrackProb>(), get<kBandwidth>());
        });
  }

  index latency()
  {
    return get<kFFT>().winSize() +
           (get<kFFT>().hopSize() * get<kMinTrackLen>());
  }
  void reset()
  {
    mSTFTBufferedProcess.reset();
    mSinesExtractor.init(get<kFFT>().winSize(), get<kFFT>().fftSize(),
                         get<kFFT>().max());
  }

private:
  STFTBufferedProcess<ParamSetViewType, kFFT> mSTFTBufferedProcess;
  algorithm::SineExtraction                   mSinesExtractor;
  ParameterTrackChanges<index, index, double> mTrackValues;
};

} // namespace sines
using RTSinesClient = ClientWrapper<sines::SinesClient>;

auto constexpr NRTSineParams = makeNRTParams<sines::SinesClient>(
    InputBufferParam("source", "Source Buffer"),
    BufferParam("sines", "Sines Buffer"),
    BufferParam("residual", "Residual Buffer"));

using NRTSinesClient =
    NRTStreamAdaptor<sines::SinesClient, decltype(NRTSineParams), NRTSineParams,
                     1, 2>;

using NRTThreadedSinesClient = NRTThreadingAdaptor<NRTSinesClient>;

} // namespace client
} // namespace fluid
