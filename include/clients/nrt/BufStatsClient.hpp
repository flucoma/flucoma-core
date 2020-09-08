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
#include "../../algorithms/public/MultiStats.hpp"

namespace fluid {
namespace client {

class BufferStatsClient : public FluidBaseClient,
                          public OfflineIn,
                          public OfflineOut
{

  enum BufferStatsParamIndex {
    kSource,
    kOffset,
    kNumFrames,
    kStartChan,
    kNumChans,
    kStats,
    kNumDerivatives,
    kLow,
    kMiddle,
    kHigh,
    kOutliersCutoff,
    kWeights,
    kWeightsThreshold
  };

public:
  FLUID_DECLARE_PARAMS(InputBufferParam("source", "Source Buffer"),
                       LongParam("startFrame", "Source Offset", 0, Min(0)),
                       LongParam("numFrames", "Number of Frames", -1),
                       LongParam("startChan", "Start Channel", 0, Min(0)),
                       LongParam("numChans", "Number of Channels", -1),
                       BufferParam("stats", "Stats Buffer"),
                       LongParam("numDerivs", "Number of Derivatives", 0,
                                 Min(0), Max(2)),
                       FloatParam("low", "Low Percentile", 0, Min(0), Max(100),
                                  UpperLimit<kMiddle>()),
                       FloatParam("middle", "Middle Percentile", 50, Min(0),
                                  Max(100), LowerLimit<kLow>(),
                                  UpperLimit<kHigh>()),
                       FloatParam("high", "High Percentile", 100, Min(0),
                                  Max(100), LowerLimit<kMiddle>()),
                       FloatParam("outliersCutoff", "Outliers Cutoff", -1, Min(-1)),
                       BufferParam("weights", "Weights Buffer"),
                       FloatParam("weightsThreshold", "Weights Threshold", 0)
                        );

  BufferStatsClient(ParamSetViewType& p) : mParams(p) {}

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

    index  outputSize = processor.numStats() * (get<kNumDerivatives>() + 1);
    Result resizeResult =
        dest.resize(outputSize, numChannels, source.sampleRate());

    if (!resizeResult.ok()) return resizeResult;

    processor.init(get<kNumDerivatives>(), get<kLow>(), get<kMiddle>(),
                   get<kHigh>());

    RealVector weights;
    if (get<kWeights>()){
      BufferAdaptor::ReadAccess weightsBuf(get<kWeights>().get());
      if (!weightsBuf.exists())
        return {Result::Status::kError, "Weights buffer supplied but invalid"};
      if(weightsBuf.numChans() != 1) return {Result::Status::kError, "Weights buffer invalid channel count"};
      if(weightsBuf.numFrames() != numFrames)
        return {Result::Status::kError, "Weights buffer invalid size"};
      weights = RealVector(numFrames);
      weights = weightsBuf.samps(0);//copy from weights buffer
    }

    FluidTensor<double, 2> tmp(numChannels, numFrames);
    FluidTensor<double, 2> result(numChannels, outputSize);
    for (int i = 0; i < numChannels; i++){
      tmp.row(i) = source.samps(get<kOffset>(), numFrames, get<kStartChan>() + i);
    }
    processor.process(tmp, result, get<kOutliersCutoff>(), weights, get<kWeightsThreshold>());

    for (int i = 0; i < numChannels; i++){
      dest.samps(i) = result.row(i);
    }

    return {Result::Status::kOk, ""};
  }
};

using NRTThreadedBufferStatsClient =
    NRTThreadingAdaptor<ClientWrapper<BufferStatsClient>>;

} // namespace client
} // namespace fluid
