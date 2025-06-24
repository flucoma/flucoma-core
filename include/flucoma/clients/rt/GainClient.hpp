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
#include "../common/FluidBaseClient.hpp"
#include "../common/FluidNRTClientWrapper.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterSet.hpp"
#include "../../data/TensorTypes.hpp"


namespace fluid {
namespace client {
namespace gain {

enum GainParamTags { kGain };

constexpr auto GainParams = defineParameters(FloatParam("gain", "Gain", 1.0));

class GainClient : public FluidBaseClient, public AudioIn, public AudioOut
{
public:
  using ParamDescType = decltype(GainParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors() { return GainParams; }

  GainClient(ParamSetViewType& p, FluidContext&) : mParams(p)
  {
    audioChannelsIn(2);
    audioChannelsOut(1);
  }

  index latency() const { return 0; }
  void  reset() {}

  template <typename T>
  void process(std::vector<FluidTensorView<T, 1>>& input,
               std::vector<FluidTensorView<T, 1>>& output, FluidContext&)
  {
    // Data is stored with samples laid out in rows, one channel per row
    if (!input[0].data()) return;

    // Copy the input samples
    output[0] <<= input[0];

    // 2nd input? -> ar version
    if (input[1].data())
    {
      // Apply gain from the second channel
      output[0].apply(input[1], [](T& x, T& y) { x *= y; });
    }
    else
    {
      double g = get<kGain>();
      output[0].apply([g](T& x) { x *= static_cast<T>(g); });
    }
  }
}; // class
} // namespace gain

using RTGainClient = ClientWrapper<gain::GainClient>;

auto constexpr NRTGainParams = makeNRTParams<gain::GainClient>(
    InputBufferParam("source", "Source Buffer"),
    InputBufferParam("gainbuffer", "Gain Buffer"),
    BufferParam("out", "output Buffer"));

using NRTGainClient =
    NRTStreamAdaptor<gain::GainClient, decltype(NRTGainParams), NRTGainParams,
                     2, 1>;

using NRTThreadedGainClient = NRTThreadingAdaptor<NRTGainClient>;


} // namespace client
} // namespace fluid
