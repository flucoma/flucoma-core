#pragma once

#include <algorithms/public/EnvelopeSegmentation.hpp>
#include <clients/common/AudioClient.hpp>
#include <clients/common/FluidBaseClient.hpp>
#include <clients/common/ParameterConstraints.hpp>
#include <clients/common/ParameterSet.hpp>
#include <clients/common/ParameterTrackChanges.hpp>
#include <clients/common/ParameterTypes.hpp>
#include <clients/nrt/FluidNRTClientWrapper.hpp>
#include <clients/rt/BufferedProcess.hpp>
#include <tuple>

namespace fluid {
namespace client {

enum AmpSliceParamIndex {
  kAbsRampUpTime,
  kAbsRampDownTime,
  kAbsOnThreshold,
  kAbsOffThreshold,
  kMinTimeAboveThreshold,
  kMinEventDuration,
  kUpwardLookupTime,
  kMinTimeBelowThreshold,
  kMinSilencetDuration,
  kDownwardLookupTime,
  kRelRampUpTime,
  kRelRampDownTime,
  kRelOnThreshold,
  kRelOffThreshold,
  kHiPassFreq,
  kMaxSize,
  kOutput
};

auto constexpr AmpSliceParams = defineParameters(
    FloatParam("absRampUp", "Absolute envelope ramp up time", 10, Min(1)),
    FloatParam("absRampDown", "Absolute envelope ramp down time", 10, Min(1)),
    FloatParam("absOnThreshold", "Absolute envelope On threshold", -40,
               Min(-144), Max(0)),
    FloatParam("absOffThreshold", "Absolute envelope Off threshold", -40,
               Min(-144), Max(0)),
    LongParam("minTimeAboveThreshold",
              "Absolute envelope min time above threshold", 440, Min(0)),
    LongParam("minEventDuration", "Absolute envelope min event duration", 0,
              Min(0)),
    LongParam("upwardLookupTime", "Absolute envelope upward lookup time", 0,
              Min(0)),
    LongParam("minTimeBelowThreshold",
              "Absolute envelope min time below threshold", 440, Min(0)),
    LongParam("minSilenceDuration", "Absolute envelope min silence duration", 0,
              Min(0)),
    LongParam("downwardLookupTime", "Absolute envelope downward lookup time", 0,
              Min(0)),
    FloatParam("relRampUp", "Relative envelope ramp up time", 10, Min(1)),
    FloatParam("relRampDown", "Relative envelope ramp down time", 10, Min(1)),
    FloatParam("relOnThreshold", "Relative envelope On threshold", -40,
               Min(-144), Max(0)),
    FloatParam("relOffThreshold", "Relative envelope Off threshold", -40,
               Min(-144), Max(0)),
    FloatParam("hiPassFreq", "Hi-pass filter cutoff", 250,
               Min(1)),
    LongParam("maxSize", "Maximum latency", 441000, Min(0)),//TODO
    LongParam("outputType", "Output Type (tempiorarily)", 0, Min(0))
);

template <typename T>
class AmpSlice : public FluidBaseClient<decltype(AmpSliceParams), AmpSliceParams>,
                 public AudioIn,
                 public AudioOut {
  using HostVector = HostVector<T>;

public:
  AmpSlice(ParamSetViewType &p) : FluidBaseClient(p) {
    FluidBaseClient::audioChannelsIn(1);
    FluidBaseClient::audioChannelsOut(1);
  }

  void process(std::vector<HostVector> &input,
               std::vector<HostVector> &output) {

    if (!input[0].data() || !output[0].data())
      return;
    size_t hostVecSize = input[0].size();

    if (mTrackValues.changed(
      get<kAbsRampUpTime>(),
      get<kAbsRampDownTime>(),
      get<kAbsOnThreshold>(),
      get<kAbsOffThreshold>(),
      get<kMinTimeAboveThreshold>(),
      get<kMinEventDuration>(),
      get<kUpwardLookupTime>(),
      get<kMinTimeBelowThreshold>(),
      get<kMinSilencetDuration>(),
      get<kDownwardLookupTime>(),
      get<kRelRampUpTime>(),
      get<kRelRampDownTime>(),
      get<kRelOnThreshold>(),
      get<kRelOffThreshold>(),
      get<kHiPassFreq>()))
      {
      mAlgorithm.init(get<kHiPassFreq>(), get<kAbsRampUpTime>(),
                      get<kRelRampUpTime>(), get<kAbsRampDownTime>(),
                      get<kRelRampDownTime>(), get<kAbsOnThreshold>(),
                      get<kRelOnThreshold>(), get<kRelOffThreshold>(),
                      get<kMinTimeAboveThreshold>(), get<kMinEventDuration>(),
                      get<kUpwardLookupTime>(), get<kAbsOffThreshold>(),
                      get<kMinTimeBelowThreshold>(),
                      get<kMinSilencetDuration>(), get<kDownwardLookupTime>());

    }

    for(int i = 0; i < input[0].size(); i++){
      output[0](i) = mAlgorithm.processSample(input[0](i));
    }
  }

  size_t latency() {
    return mAlgorithm.getLatency();
  }

private:
  ParameterTrackChanges<double, double, double, double, size_t, size_t, size_t,
                        size_t, size_t, size_t, double, double, double, double,
                        double>
      mTrackValues;
  algorithm::EnvelopeSegmentation mAlgorithm{get<kMaxSize>(), get<kOutput>()};
};

auto constexpr NRTAmpSliceParams =
    makeNRTParams<AmpSlice>({BufferParam("source", "Source Buffer")},
                                   {BufferParam("indices", "Indices Buffer")});

template <typename T>
using NRTAmpSlice = NRTSliceAdaptor<AmpSlice<T>, decltype(NRTAmpSliceParams),
                                    NRTAmpSliceParams, 1, 1>;
} // namespace client
} // namespace fluid
