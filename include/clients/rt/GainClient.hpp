/*
 @file GainClient.hpp

 Simple multi-input client, just does modulation of signal 1 by signal 2, or
 scalar gain change
 */
#pragma once

#include <clients/common/AudioClient.hpp>
#include <clients/common/FluidBaseClient.hpp>
#include <clients/common/ParameterConstraints.hpp>
#include <clients/common/ParameterSet.hpp>
#include <data/TensorTypes.hpp>

namespace fluid {
namespace client {

enum GainParamTags { kGain };

constexpr auto GainParams = defineParameters(FloatParam("gain", "Gain", 1.0));

/// @class GainAudioClient
template <typename T>
class GainClient : public FluidBaseClient<decltype(GainParams), GainParams>, public AudioIn, public AudioOut
{
  using HostVector = FluidTensorView<T,1>;
public:
  GainClient(ParamSetType &p) : FluidBaseClient(p) {
    FluidBaseClient::audioChannelsIn(2);
    FluidBaseClient::audioChannelsOut(1);
  }

  size_t latency() { return 0; }

  void process(std::vector<HostVector> &input, std::vector<HostVector> &output)
  {
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
} // namespace client
} // namespace fluid
