/*
 @file GainClient.hpp

 Simple multi-input client, just does modulation of signal 1 by signal 2, or
 scalar gain change
 */
#pragma once

#include "../common/AudioClient.hpp"
#include "../common/FluidBaseClient.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterSet.hpp"
#include "../../data/TensorTypes.hpp"

#include "../nrt/FluidNRTClientWrapper.hpp"

namespace fluid {
namespace client {

class GainClient : public FluidBaseClient, public AudioIn, public AudioOut
{
  enum GainParamTags { kGain };
public:

  FLUID_DECLARE_PARAMS(FloatParam("gain", "Gain", 1.0));

  GainClient(ParamSetViewType &p) : mParams(p) {
    audioChannelsIn(2);
    audioChannelsOut(1);
  }

  size_t latency() { return 0; }

  template <typename T>
  void process(std::vector<FluidTensorView<T,1>> &input, std::vector<FluidTensorView<T,1>> &output, FluidContext& c,
               bool reset = false) {
    // Data is stored with samples laid out in rows, one channel per row
    if (!input[0].data())
      return;

    // Copy the input samples
    output[0] = input[0];

    // 2nd input? -> ar version
    if (input[1].data()) {
        // Apply gain from the second channel
        output[0].apply(input[1], [](T &x, T &y) { x *= y; });
    } else {
      double g = get<kGain>();
      output[0].apply([g](T &x) { x *= g; });
    }
  }
}; // class

using RTGainClient = ClientWrapper<GainClient>;

auto constexpr NRTGainParams = makeNRTParams<GainClient>({InputBufferParam("source", "Source Buffer"),InputBufferParam("gainbuffer","Gain Buffer")},  {BufferParam("out","output Buffer")});

using NRTGain = NRTStreamAdaptor<RTGainClient, decltype(NRTGainParams), NRTGainParams, 2, 1>;
  
using NRTThreadedGain = NRTThreadingAdaptor<NRTGain>;


} // namespace client
} // namespace fluid
