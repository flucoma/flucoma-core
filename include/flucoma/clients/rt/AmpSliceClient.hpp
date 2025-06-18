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
#include "../../algorithms/public/EnvelopeSegmentation.hpp"
#include <tuple>

namespace fluid {
namespace client {
namespace ampslice {

enum AmpSliceParamIndex {
  kFastRampUpTime,
  kFastRampDownTime,
  kSlowRampUpTime,
  kSlowRampDownTime,
  kOnThreshold,
  kOffThreshold,
  kSilenceThreshold,
  kDebounce,
  kHiPassFreq,
};

constexpr auto AmpSliceParams = defineParameters(
    LongParam("fastRampUp", "Fast Envelope Ramp Up Length", 1, Min(1)),
    LongParam("fastRampDown", "Fast Envelope Ramp Down Length", 1, Min(1)),
    LongParam("slowRampUp", "Slow Envelope Ramp Up Length", 100, Min(1)),
    LongParam("slowRampDown", "Slow Envelope Ramp Down Length", 100, Min(1)),
    FloatParam("onThreshold", "On Threshold (dB)", 144, Min(-144), Max(144)),
    FloatParam("offThreshold", "Off Threshold (dB)", -144, Min(-144), Max(144)),
    FloatParam("floor", "Floor value (dB)", -144, Min(-144), Max(144)),
    LongParam("minSliceLength", "Minimum Length of Slice", 2, Min(0)),
    FloatParam("highPassFreq", "High-Pass Filter Cutoff", 85, Min(0)));

class AmpSliceClient : public FluidBaseClient, public AudioIn, public AudioOut
{

public:
  using ParamDescType = decltype(AmpSliceParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors() { return AmpSliceParams; }

  AmpSliceClient(ParamSetViewType& p, FluidContext&) : mParams(p)
  {
    audioChannelsIn(1);
    audioChannelsOut(1);
    FluidBaseClient::setInputLabels({"audio input"});
    FluidBaseClient::setOutputLabels({"1 when slice detected, 0 otherwise"});
  }

  template <typename T>
  void process(std::vector<HostVector<T>>& input,
               std::vector<HostVector<T>>& output, FluidContext&)
  {

    if (!input[0].data() || !output[0].data()) return;

    double hiPassFreq = std::min(get<kHiPassFreq>() / sampleRate(), 0.5);

    if (!mAlgorithm.initialized())
    { mAlgorithm.init(get<kSilenceThreshold>(), hiPassFreq); }
    for (index i = 0; i < input[0].size(); i++)
    {
      output[0](i) = static_cast<T>(mAlgorithm.processSample(
          input[0](i), get<kOnThreshold>(), get<kOffThreshold>(),
          get<kSilenceThreshold>(), get<kFastRampUpTime>(),
          get<kSlowRampUpTime>(), get<kFastRampDownTime>(),
          get<kSlowRampDownTime>(), hiPassFreq, get<kDebounce>()));
    }
  }
  index latency() const { return 0; }

  void reset(FluidContext&)
  {
    double hiPassFreq = std::min(get<kHiPassFreq>() / sampleRate(), 0.5);
    mAlgorithm.init(get<kSilenceThreshold>(), hiPassFreq);
  }

private:
  algorithm::EnvelopeSegmentation mAlgorithm;
};
} // namespace ampslice

using RTAmpSliceClient = ClientWrapper<ampslice::AmpSliceClient>;

auto constexpr NRTAmpSliceParams = makeNRTParams<ampslice::AmpSliceClient>(
    InputBufferParam("source", "Source Buffer"),
    BufferParam("indices", "Indices Buffer"));

using NRTAmpSliceClient =
    NRTSliceAdaptor<ampslice::AmpSliceClient, decltype(NRTAmpSliceParams),
                    NRTAmpSliceParams, 1, 1>;

using NRTThreadedAmpSliceClient = NRTThreadingAdaptor<NRTAmpSliceClient>;

} // namespace client
} // namespace fluid
