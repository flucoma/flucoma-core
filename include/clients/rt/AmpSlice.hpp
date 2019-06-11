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
  kMinEventDuration,
  kMinSilencetDuration,
  kMinTimeAboveThreshold,
  kMinTimeBelowThreshold,
  kUpwardLookupTime,
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
    FloatParam("absRampUp", "Absolute Envelope Ramp Up Length", 10, Min(1)),
    FloatParam("absRampDown", "Absolute Envelope Ramp Down Length", 10, Min(1)),
    FloatParam("absThreshOn", "Absolute Envelope Threshold On", -40, Min(-144), Max(144)),
    FloatParam("absThreshOff", "Absolute Envelope Threshold Off", -40, Min(-144), Max(144)),
    LongParam("minSliceLength", "Minimum Length of Slice", 1, Min(1)),
    LongParam("minSilenceLength", "Absolute Envelope Minimum Length of Silence", 1, Min(1)),
    LongParam("minLengthAbove","Required Minimal Length Above Threshold", 1, Min(1)),
    LongParam("minLengthBelow", "Required Minimal Length Below Threshold", 1, Min(1)),
    LongParam("lookBack", "Absolute Envelope Backward Lookup Length", 0, Min(0)),
    LongParam("lookAhead", "Absolute Envelope Forward Lookup Length", 0, Min(0)),
    FloatParam("relRampUp", "Relative Envelope Ramp Up Length", 1, Min(1)),
    FloatParam("relRampDown", "Relative Envelope Ramp Down Length", 1, Min(1)),
    FloatParam("relThreshOn", "Relative Envelope Threshold On", -144 , Min(-144), Max(144)),
    FloatParam("relThreshOff", "Relative Envelope Threshold Off", -144, Min(-144), Max(144)),
    FloatParam("highPassFreq", "High-Pass Filter Cutoff", 250, Min(1)),
    LongParam<Fixed<true>>("maxSize", "Maximum Total Latency", 88200, Min(1)),//TODO
    LongParam("outputType", "Output Type (temporarily)", 0, Min(0))
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
               std::vector<HostVector> &output, bool reset = false) {

    if (!input[0].data() || !output[0].data())
      return;
    size_t hostVecSize = input[0].size();

    if (!mAlgorithm.initialized() || mTrackValues.changed(
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
      double hiPassFreq = get<kHiPassFreq>() / sampleRate();
      if (hiPassFreq >1) hiPassFreq = 1;
      mAlgorithm.init(hiPassFreq, get<kAbsRampUpTime>(),
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
