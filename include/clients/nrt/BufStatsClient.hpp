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

#include "../common/FluidBaseClient.hpp"
#include "../common/FluidNRTClientWrapper.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterTypes.hpp"
#include "../../algorithms/public/Stats.hpp"

namespace fluid {
namespace client {

enum BufStatsParamIndex {
  kSource,
  kOffset,
  kNumFrames,
  kStartChan,
  kNumChans,
  kStats,
  kNumDerivatives,
  kLow,
  kMiddle,
  kHigh
};

auto constexpr BufStatsParams = defineParameters(
    InputBufferParam("source", "Source Buffer"),
    LongParam("startFrame", "Source Offset", 0, Min(0)),
    LongParam("numFrames", "Number of Frames", -1),
    LongParam("startChan", "Start Channel", 0, Min(0)),
    LongParam("numChans", "Number of Channels", -1),
    BufferParam("stats", "Stats Buffer"),
    LongParam("numDerivs", "Number of Derivatives", 0, Min(0), Max(2)),
    FloatParam("low", "Low Percentile", 0, Min(0), Max(100),
               UpperLimit<kMiddle>()),
    FloatParam("middle", "Middle Percentile", 50, Min(0), Max(100),
               LowerLimit<kLow>(), UpperLimit<kHigh>()),
    FloatParam("high", "High Percentile", 100, Min(0), Max(100),
               LowerLimit<kMiddle>()));

template <typename T>
class BufStatsClient
    : public FluidBaseClient<decltype(BufStatsParams), BufStatsParams>,
      public OfflineIn,
      public OfflineOut
{

public:
  BufStatsClient(ParamSetViewType& p) : FluidBaseClient(p) {}

  Result process(FluidContext& c)
  {
    algorithm::Stats processor;

    if (!get<kSource>().get())
      return {Result::Status::kError, "No input buffer supplied"};

    if (!get<kStats>().get())
      return {Result::Status::kError, "No output buffer supplied"};

    BufferAdaptor::ReadAccess source(get<kSource>().get());
    BufferAdaptor::Access     dest(get<kStats>().get());

    if (!source.exists())
      return {Result::Status::kError, "Input buffer not found"};

    if (!source.valid())
      return {Result::Status::kError, "Can't access input buffer"};

    if (!dest.exists())
      return {Result::Status::kError, "Output buffer not found"};

    if (get<kOffset>() >= source.numFrames())
      return {Result::Status::kError, "Start frame (", get<kOffset>(),
              ") out of range."};

    index numFrames = get<kNumFrames>() == -1
                          ? (source.numFrames() - get<kOffset>())
                          : get<kNumFrames>();
    index numChannels = get<kNumChans>() == -1
                            ? (source.numChans() - get<kStartChan>())
                            : get<kNumChans>();

    if (get<kOffset>() + numFrames > source.numFrames())
      return {Result::Status::kError, "Start frame + num frames (",
              get<kOffset>() + get<kNumFrames>(), ") out of range."};

    if (get<kStartChan>() + numChannels > source.numChans())
      return {Result::Status::kError, "Start channel ", get<kStartChan>(),
              " out of range."};

    if (numChannels <= 0 || numFrames <= 0)
      return {Result::Status::kError, "Zero length segment requested"};

    if (numFrames <= get<kNumDerivatives>())
      return {Result::Status::kError, "Not enough frames"};

    index  outputSize = processor.numStats() * (get<kNumDerivatives>() + 1);
    Result resizeResult =
        dest.resize(outputSize, numChannels, source.sampleRate());

    if (!resizeResult.ok()) return resizeResult;

    processor.init(get<kNumDerivatives>(), get<kLow>(), get<kMiddle>(),
                   get<kHigh>());
    for (int i = 0; i < numChannels; i++)
    {
      auto sourceChannel = FluidTensor<double, 1>(numFrames);
      auto destChannel = FluidTensor<double, 1>(outputSize);
      for (int j = 0; j < numFrames; j++)
        sourceChannel(j) =
            source.samps(get<kOffset>(), numFrames, get<kStartChan>() + i)(j);
      processor.process(sourceChannel, destChannel);

      if (c.task() && !c.task()->processUpdate(i + 1, numChannels))
        return {Result::Status::kCancelled, ""};

      for (index j = 0; j < outputSize; j++) dest.samps(i)(j) = static_cast<float>(destChannel(j));
    }

    return {Result::Status::kOk, ""};
  }
};

template <typename T>
using NRTThreadedBufStatsClient = NRTThreadingAdaptor<BufStatsClient<T>>;

} // namespace client
} // namespace fluid
