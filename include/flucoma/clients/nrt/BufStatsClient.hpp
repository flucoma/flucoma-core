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

#include "../common/FluidBaseClient.hpp"
#include "../common/FluidNRTClientWrapper.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterTypes.hpp"
#include "../../algorithms/public/MultiStats.hpp"
#include <algorithm>

namespace fluid {
namespace client {
namespace bufstats {

enum BufferStatsParamIndex {
  kSource,
  kOffset,
  kNumFrames,
  kStartChan,
  kNumChans,
  kStats,
  kSelect,
  kNumDerivatives,
  kLow,
  kMiddle,
  kHigh,
  kOutliersCutoff,
  kWeights
};

constexpr auto BufStatsParams = defineParameters(
    InputBufferParam("source", "Source Buffer"),
    LongParam("startFrame", "Source Offset", 0, Min(0)),
    LongParam("numFrames", "Number of Frames", -1),
    LongParam("startChan", "Start Channel", 0, Min(0)),
    LongParam("numChans", "Number of Channels", -1),
    BufferParam("stats", "Stats Buffer"),
    ChoicesParam("select","Selection of Statistics","mean","std","skew","kurtosis","low","mid","high"),
    LongParam("numDerivs", "Number of Derivatives", 0, Min(0), Max(2)),
    FloatParam("low", "Low Percentile", 0, Min(0), Max(100),
               UpperLimit<kMiddle>()),
    FloatParam("middle", "Middle Percentile", 50, Min(0), Max(100),
               LowerLimit<kLow>(), UpperLimit<kHigh>()),
    FloatParam("high", "High Percentile", 100, Min(0), Max(100),
               LowerLimit<kMiddle>()),
    FloatParam("outliersCutoff", "Outliers Cutoff", -1, Min(-1)),
    InputBufferParam("weights", "Weights Buffer"));

class BufferStatsClient : public FluidBaseClient,
                          public OfflineIn,
                          public OfflineOut
{
public:
  using ParamDescType = decltype(BufStatsParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors() { return BufStatsParams; }


  BufferStatsClient(ParamSetViewType& p, FluidContext&) : mParams(p) {}

  template <typename T>
  Result process(FluidContext&)
  {
    algorithm::MultiStats processor;

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

    index  outputSize = asSigned(get<kSelect>().count()) * (get<kNumDerivatives>() + 1);
    index  processorOutputSize = processor.numStats() * (get<kNumDerivatives>() + 1);
    
    Result resizeResult =
        dest.resize(outputSize, numChannels, source.sampleRate());

    if (!resizeResult.ok()) return resizeResult;

    processor.init(get<kNumDerivatives>(), get<kLow>(), get<kMiddle>(),
                   get<kHigh>());

    Result     processingResult = {Result::Status::kOk, ""};
    RealVector weights;

    if (get<kWeights>())
    {
      BufferAdaptor::ReadAccess weightsBuf(get<kWeights>().get());
      if (!weightsBuf.exists())
        return {Result::Status::kError, "Weights buffer supplied but invalid"};
      if (weightsBuf.numChans() != 1)
        return {Result::Status::kError, "Weights buffer invalid channel count"};
      if (weightsBuf.numFrames() != numFrames)
        return {Result::Status::kError, "Weights buffer invalid size"};
      weights = RealVector(numFrames);
      weights <<= weightsBuf.samps(0); // copy from buffer
      if (*std::min_element(weights.begin(), weights.end()) < 0)
      {
        processingResult =
            Result(Result::Status::kWarning, "Negative weights clipped to 0");
      }
      if (*std::max_element(weights.begin(), weights.end()) <= 0)
      {
        for (index i = 0; i < numChannels; i++) dest.samps(i).fill(0);
        return {Result::Status::kWarning, "Invalid weights"};
      }
    }
    FluidTensor<double, 2> tmp(numChannels, numFrames);
    FluidTensor<double, 2> result(numChannels, processorOutputSize);
    for (int i = 0; i < numChannels; i++)
    {
      tmp.row(i) <<=
          source.samps(get<kOffset>(), numFrames, get<kStartChan>() + i);
    }
    processor.process(tmp, result, get<kOutliersCutoff>(), weights);
    
    auto selection = get<kSelect>();
    
    for (index i = 0; i < numChannels; ++i)
    {
      auto outputChannel = dest.samps(i);
      auto resultChannel = result.row(i);
      for(index j = 0, k = 0; j < processorOutputSize; ++j)
      {
         if(selection[j % 7])
          outputChannel(k++) = static_cast<T>(resultChannel(j));
      }
    }
    
    return processingResult;
  }
};
} // namespace bufstats

using NRTThreadedBufferStatsClient =
    NRTThreadingAdaptor<ClientWrapper<bufstats::BufferStatsClient>>;

} // namespace client
} // namespace fluid
