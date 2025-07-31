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
#include "../../algorithms/public/SpectralShape.hpp"
#include "../../data/TensorTypes.hpp"
#include <tuple>

namespace fluid {
namespace client {
namespace spectralshape {

using algorithm::SpectralShape;

enum SpectralShapeParamIndex {
  kSelect,
  kMinFreq,
  kMaxFreq,
  kRollOffPercent,
  kFreqUnits,
  kAmpMeasure,
  kFFT
};

constexpr auto SpectralShapeParams = defineParameters(
    ChoicesParam("select", "Selection of Features", "centroid", "spread",
                 "skew", "kurtosis", "rolloff", "flatness", "crest"),
    FloatParam("minFreq", "Low Frequency Bound", 0, Min(0)),
    FloatParam("maxFreq", "High Frequency Bound", -1, Min(-1)),
    FloatParam("rolloffPercent", "Rolloff Percent", 95, Min(0), Max(100)),
    EnumParam("unit", "Frequency Unit", 0, "Hz", "Midi Cents"),
    EnumParam("power", "Use Power", 0, "No", "Yes"),
    FFTParam("fftSettings", "FFT Settings", 1024, -1, -1));

class SpectralShapeClient : public FluidBaseClient,
                            public AudioIn,
                            public ControlOut
{
  static constexpr index mMaxOutputSize = 7;

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

  SpectralShapeClient(ParamSetViewType& p, FluidContext& c)
      : mParams(p), mSTFTBufferedProcess(get<kFFT>(), 1, 0, c.hostVectorSize(),
                                         c.allocator()),
        mAlgorithm{c.allocator()},
        mMagnitude(get<kFFT>().maxFrameSize(), c.allocator()),
        mDescriptors(7, c.allocator())
  {
    audioChannelsIn(1);
    controlChannelsOut({1, asSigned(get<kSelect>().count()), mMaxOutputSize});
    setInputLabels({"audio input"});
    setOutputLabels({"spectral features"});
    mDescriptors = FluidTensor<double, 1>(mMaxOutputSize);
  }

  template <typename T>
  void process(std::vector<HostVector<T>>& input,
               std::vector<HostVector<T>>& output, FluidContext& c)
  {
    using std::size_t;

    if (!input[0].data() || !output[0].data()) return;
    assert(controlChannelsOut().size && "No control channels");
    assert(output[0].size() >= controlChannelsOut().size &&
           "Too few output channels");

    if (mHostSizeTracker.changed(c.hostVectorSize()))
    {
      mSTFTBufferedProcess = STFTBufferedProcess<>(get<kFFT>(), 1, 0, c.hostVectorSize(),
                                         c.allocator());
    }

    mSTFTBufferedProcess.processInput(
        get<kFFT>(), input, c, [&](ComplexMatrixView in) {
          algorithm::STFT::magnitude(in.row(0),
                                     mMagnitude(Slice(0, in.size())));
          mAlgorithm.processFrame(
              mMagnitude(Slice(0, in.size())), mDescriptors, sampleRate(),
              get<kMinFreq>(), get<kMaxFreq>(), get<kRollOffPercent>(),
              get<kFreqUnits>() == 1, get<kAmpMeasure>() == 1, c.allocator());
        });

    auto  selection = get<kSelect>();
    index numSelected = asSigned(selection.count());
    index numOuts = std::min<index>(mMaxOutputSize, numSelected);
    controlChannelsOut({1, numOuts, mMaxOutputSize});

    for (index i = 0, j = 0; i < mMaxOutputSize && j < numOuts; ++i)
    {
      if (selection[asUnsigned(i)])
        output[0](j++) = static_cast<T>(mDescriptors(i));
    }

    output[0](Slice(numOuts, mMaxOutputSize - numOuts)).fill(0);
  }

  index latency() const { return get<kFFT>().winSize(); }

  void reset(FluidContext&) { mSTFTBufferedProcess.reset(); }

  AnalysisSize analysisSettings()
  {
    return {get<kFFT>().winSize(), get<kFFT>().hopSize()};
  }

private:
  ParameterTrackChanges<index>         mHostSizeTracker;
  STFTBufferedProcess<>                mSTFTBufferedProcess;

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
