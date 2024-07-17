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

#include "../common/AudioClient.hpp"
#include "../common/FluidBaseClient.hpp"
#include "../common/FluidNRTClientWrapper.hpp"
#include "../common/FluidSource.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/ParameterTypes.hpp"
#include "../../data/TensorTypes.hpp"
#include "../../algorithms/public/VoiceAllocator.hpp"

namespace fluid {
namespace client {
namespace voiceallocator {

template <typename T>
using HostVector = FluidTensorView<T, 1>;

enum VoiceAllocatorParamIndex {
  kNVoices,
  kPrioritisedVoices,
  kBirthLowThreshold,
  kBirthHighTreshold,
  kMinTrackLen,
  kTrackMagRange,
  kTrackFreqRange,
  kTrackProb
};

constexpr auto VoiceAllocatorParams = defineParameters(
    LongParamRuntimeMax<Primary>( "numVoices", "Number of Voices", 1, Min(1)),
    EnumParam("prioritisedVoices", "Prioritised Voice Quality", 0, "Lowest Frequency", "Loudest Magnitude"),
    FloatParam("birthLowThreshold", "Track Birth Low Frequency Threshold", -24, Min(-144), Max(0)),
    FloatParam("birthHighThreshold", "Track Birth High Frequency Threshold", -60, Min(-144), Max(0)),
    LongParam("minTrackLen", "Minimum Track Length", 1, Min(1)),
    FloatParam("trackMagRange", "Tracking Magnitude Range (dB)", 15., Min(1.), Max(200.)),
    FloatParam("trackFreqRange", "Tracking Frequency Range (Hz)", 50., Min(1.), Max(10000.)),
    FloatParam("trackProb", "Tracking Matching Probability", 0.5, Min(0.0), Max(1.0))
    );

class VoiceAllocatorClient : public FluidBaseClient,
                             public ControlIn,
                             ControlOut
{
    using VoicePeak = algorithm::VoicePeak;
    using SinePeak = algorithm::SinePeak;

public:
  using ParamDescType = decltype(VoiceAllocatorParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors()
  {
    return VoiceAllocatorParams;
  }

  VoiceAllocatorClient(ParamSetViewType& p, FluidContext& c)
    : mParams(p), mVoiceAllocator(get<kNVoices>().max(), c.allocator()),
    mSizeTracker{ 0 }
  {
    controlChannelsIn(2);
    controlChannelsOut({3, get<kNVoices>(), get<kNVoices>().max()});
    setInputLabels({"frequencies", "magnitudes"});
    setOutputLabels({"frequencies", "magnitudes", "states"});
    mVoiceAllocator.init(get<kNVoices>(), c.allocator());
  }

  template <typename T>
  void process(std::vector<HostVector<T>>& input,
               std::vector<HostVector<T>>& output, FluidContext& c)
  {
    if (!input[0].data()) return;
    if (!output[0].data() && !output[1].data()) return;
    if (!mVoiceAllocator.initialized() || mSizeTracker.changed(get<kNVoices>()))
    {
      controlChannelsOut({ 4, get<kNVoices>() }); //update the dynamic out size
      mVoiceAllocator.init(get<kNVoices>(), c.allocator());
    }

    rt::vector<SinePeak> incomingVoices(0, c.allocator());
    rt::vector<VoicePeak> outgoingVoices(0, c.allocator());

    for (index i = 0; i < input[0].size(); ++i)
    {
        if (input[1].row(i) != 0 && input[0].row(i) != 0)
        {
            double logMag = 20 * log10(std::max(static_cast<double>(input[1].row(i)), algorithm::epsilon));
            incomingVoices.push_back({ input[0].row(i), logMag, false });
        }
    }

    mVoiceAllocator.processFrame(incomingVoices, outgoingVoices, get<kMinTrackLen>(), get<kBirthLowThreshold>(), get<kBirthHighTreshold>(), 0, get<kTrackMagRange>(), get<kTrackFreqRange>(), get<kTrackProb>(), get<kPrioritisedVoices>(), c.allocator());

    for (index i = 0; i < static_cast<index>(get<kNVoices>()); ++i)
    {
        output[2].row(i) = static_cast<index>(outgoingVoices[i].state);
        output[1].row(i) = outgoingVoices[i].logMag;
        output[0].row(i) = outgoingVoices[i].freq;
    }
  }

  MessageResult<void> clear()
  {
    mVoiceAllocator.reset();
    return {};
  }

    void reset(FluidContext&)
    {
        clear();
    }

  static auto getMessageDescriptors()
  {
    return defineMessages(makeMessage("clear", &VoiceAllocatorClient::clear));
  }

  index latency() const { return 0; }

private:
    algorithm::VoiceAllocator                   mVoiceAllocator;
    ParameterTrackChanges<index>                mSizeTracker;
};

} // namespace voiceallocator

using VoiceAllocatorClient =
    ClientWrapper<voiceallocator::VoiceAllocatorClient>;

auto constexpr NRTVoiceAllocatorParams = makeNRTParams<voiceallocator::VoiceAllocatorClient>(
    InputBufferParam("frequencies", "Source F Buffer"),
    InputBufferParam("magnitudes", "Source M Buffer"),
    BufferParam("freqed", "dest f Buffer"),
    BufferParam("magned", "dest m Buffer"),
    BufferParam("voiced", "dest v Buffer")
    );

using NRTVoiceAllocatorClient = NRTDualControlAdaptor<voiceallocator::VoiceAllocatorClient,
                           decltype(NRTVoiceAllocatorParams), NRTVoiceAllocatorParams, 2, 3>;

using NRTThreadedVoiceAllocatorClient = NRTThreadingAdaptor<NRTVoiceAllocatorClient>;
} // namespace client
} // namespace fluid
