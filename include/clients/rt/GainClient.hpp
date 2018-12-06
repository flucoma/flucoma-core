/*
 @file GainClient.hpp

 Simple multi-input client, just does modulation of signal 1 by signal 2, or
 scalar gain change
 */
#pragma once

#include <clients/common/AudioClient.hpp>
#include <clients/common/FluidBaseClient.hpp>
#include <clients/common/ParameterConstraints.hpp>

namespace fluid {
namespace client {

enum GainParamTags { kGain, kWindowSize, kMaxSize };

constexpr auto GainParams = std::make_tuple(FloatParam("gain", "Gain", 1.0));

using Params_t = decltype(GainParams);

/// @class GainAudioClient
template <typename T, typename U = T>
class GainClient : public FluidBaseClient<Params_t> {
  using View = fluid::FluidTensorView<U, 1>;

public:
  enum class Params { kGain, kMaxWindow, kWindow, kHop };

  /**
   No default instances, no copying
   **/
  GainClient(GainClient &) = delete;
  GainClient operator=(GainClient &) = delete;

  /**
   Construct with a (maximum) chunk size and some input channels
   **/
  GainClient() : FluidBaseClient<Params_t>(GainParams) {
    audioChannelsIn(2);
    audioChannelsOut(1);
  }

  /// Do the magic: we take vectors of views
  void process(std::vector<View> &input, std::vector<View> &output) {
    // Punishment crashes for the sloppy
    // Data is stored with samples laid out in rows, one channel per row
    if (!input[0].data())
      return;

    // Copy the input samples
    output[0] = input[0];

    // 2nd input? -> ar version
    if (input[1].data()) {
      output[0].apply(input[1], [](U &x, U &y) { x *= y; });
    } else {
      double g = get<kGain>();
      // Apply gain from the second channel
      output[0].apply([g](U &x) { x *= g; });
    }
  }

}; // class
} // namespace client
} // namespace fluid
