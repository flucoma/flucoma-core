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
#include "../../algorithms/public/SpectralShape.hpp"
#include "../../data/TensorTypes.hpp"
#include <tuple>

namespace fluid {
namespace client {
namespace spectralshape {

using algorithm::SpectralShape;

enum SpectralShapeParamIndex {
  kMinFreq,
  kMaxFreq,
  kRollOffPercent,
  kFreqUnits,
  kAmpMeasure,
  kFFT,
  kMaxFFTSize
};

constexpr auto SpectralShapeParams = defineParameters(
    FloatParam("minFreq", "Low Frequency Bound", 0, Min(0)),
    FloatParam("maxFreq", "High Frequency Bound", -1, Min(-1)),
    FloatParam("rolloffPercent", "Rolloff Percent", 0.95, Min(0), Max(1)),
    EnumParam("unit", "Frequency Unit", 0, "Hz", "Midi Cents"),
    EnumParam("power", "Use Power", 0, "No", "Yes"),
    FFTParam<kMaxFFTSize>("fftSettings", "FFT Settings", 1024, -1, -1),
    LongParam<Fixed<true>>("maxFFTSize", "Maxiumm FFT Size", 16384, Min(4),
                           PowerOfTwo{}));

class SpectralShapeClient : public FluidBaseClient,
                            public AudioIn,
                            public ControlOut
{
public:
  using ParamDescType = decltype(SpectralShapeParams);

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
    return SpectralShapeParams;
  }

  SpectralShapeClient(ParamSetViewType& p)
      : mParams(p), mSTFTBufferedProcess(get<kMaxFFTSize>(), 1, 0)
  {
    audioChannelsIn(1);
    controlChannelsOut(7);
    setInputLabels({"audio input"});
    setOutputLabels({"centroid, skew, skewness, kurtosis, rolloff, flatness, crest factor"});
    mDescriptors = FluidTensor<double, 1>(7);
  }

  template <typename T>
  void process(std::vector<HostVector<T>>& input,
               std::vector<HostVector<T>>& output, FluidContext& c)
  {
    using std::size_t;

    if (!input[0].data() || !output[0].data()) return;
    assert(FluidBaseClient::controlChannelsOut() && "No control channels");
    assert(output.size() >= asUnsigned(FluidBaseClient::controlChannelsOut()) &&
           "Too few output channels");

    if (mTracker.changed(get<kFFT>().frameSize(), sampleRate()))
    { mMagnitude.resize(get<kFFT>().frameSize()); }

    mSTFTBufferedProcess.processInput(
        mParams, input, c, [&](ComplexMatrixView in) {
          algorithm::STFT::magnitude(in.row(0), mMagnitude);
          mAlgorithm.processFrame(
              mMagnitude, mDescriptors, sampleRate(), get<kMinFreq>(),
              get<kMaxFreq>(), get<kRollOffPercent>(), get<kFreqUnits>() == 1,
              get<kAmpMeasure>() == 1);
        });

    for (int i = 0; i < 7; ++i)
      output[asUnsigned(i)](0) = static_cast<T>(mDescriptors(i));
  }

  index latency() { return get<kFFT>().winSize(); }

  void reset() { mSTFTBufferedProcess.reset(); }

  index controlRate() { return get<kFFT>().hopSize(); }

private:
  ParameterTrackChanges<index, double>        mTracker;
  STFTBufferedProcess<ParamSetViewType, kFFT> mSTFTBufferedProcess;

  SpectralShape          mAlgorithm;
  FluidTensor<double, 1> mMagnitude;
  FluidTensor<double, 1> mDescriptors;
};
} // namespace spectralshape

using RTSpectralShapeClient = ClientWrapper<spectralshape::SpectralShapeClient>;

auto constexpr NRTSpectralShapeParams =
    makeNRTParams<spectralshape::SpectralShapeClient>(
        InputBufferParam("source", "Source Buffer"),
        BufferParam("features", "Features Buffer"));

using NRTSpectralShapeClient =
    NRTControlAdaptor<spectralshape::SpectralShapeClient,
                      decltype(NRTSpectralShapeParams), NRTSpectralShapeParams,
                      1, 1>;

using NRTThreadedSpectralShapeClient =
    NRTThreadingAdaptor<NRTSpectralShapeClient>;

} // namespace client
} // namespace fluid
