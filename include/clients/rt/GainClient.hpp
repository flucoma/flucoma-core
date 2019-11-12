/*
Copyright 2017-2019 University of Huddersfield.
Licensed under the BSD-3 License.
See LICENSE file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/
#pragma once

#include "../common/AudioClient.hpp"
#include "../common/FluidBaseClient.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterSet.hpp"
#include "../../data/TensorTypes.hpp"

namespace fluid {
namespace client {

enum GainParamTags { kGain };

constexpr auto GainParams = defineParameters(FloatParam("gain", "Gain", 1.0));

/// @class GainAudioClient
template <typename T>
class GainClient : public FluidBaseClient<decltype(GainParams), GainParams>,
                   public AudioIn,
                   public AudioOut
{
  using HostVector = FluidTensorView<T, 1>;

public:
  GainClient(ParamSetViewType& p) : FluidBaseClient(p)
  {
    FluidBaseClient::audioChannelsIn(2);
    FluidBaseClient::audioChannelsOut(1);
  }

  size_t latency() { return 0; }

  void process(std::vector<HostVector>& input, std::vector<HostVector>& output,
               FluidContext&, bool reset = false)
  {
    // Data is stored with samples laid out in rows, one channel per row
    if (!input[0].data()) return;

    // Copy the input samples
    output[0] = input[0];

    // 2nd input? -> ar version
    if (input[1].data())
    {
      // Apply gain from the second channel
      output[0].apply(input[1], [](T& x, T& y) { x *= y; });
    }
    else
    {
      double g = get<kGain>();
      output[0].apply([g](T& x) { x *= g; });
    }
  }
}; // class
} // namespace client
} // namespace fluid
