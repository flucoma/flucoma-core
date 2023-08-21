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
#include "../common/FluidSource.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/ParameterTypes.hpp"
// #include "../../algorithms/public/RunningStats.hpp"
#include "../../data/TensorTypes.hpp"
#include "../../algorithms/util/PartialTracking.hpp"

namespace fluid {
namespace client {
namespace voiceallocator {

template <typename T>
using HostVector = FluidTensorView<T, 1>;

enum VoiceAllocatorParamIndex {
  kNVoices,
  kBirthLowThreshold,
  kBirthHighTreshold,
  kMinTrackLen,
  kTrackMethod,
  kTrackMagRange,
  kTrackFreqRange,
  kTrackProb
};

constexpr auto VoiceAllocatorParams = defineParameters(
    LongParamRuntimeMax<Primary>( "numVoices", "Number of Voices", 1, Min(1)),
    FloatParam("birthLowThreshold", "Track Birth Low Frequency Threshold", -24, Min(-144), Max(0)),
    FloatParam("birthHighThreshold", "Track Birth High Frequency Threshold", -60, Min(-144), Max(0)),
    LongParam("minTrackLen", "Minimum Track Length", 1, Min(1)),
    EnumParam("trackMethod", "Tracking Method", 0, "Greedy", "Hungarian"),
    FloatParam("trackMagRange", "Tracking Magnitude Range (dB)", 15., Min(1.), Max(200.)),
    FloatParam("trackFreqRange", "Tracking Frequency Range (Hz)", 50., Min(1.), Max(10000.)),
    FloatParam("trackProb", "Tracking Matching Probability", 0.5, Min(0.0), Max(1.0))
    );

class VoiceAllocatorClient : public FluidBaseClient,
                             public ControlIn,
                             ControlOut
{
    template <typename T>
    using vector = rt::vector<T>;
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
      : mParams(p), mTracking(c.allocator()),
      mInputSize{ 0 }, mSizeTracker{ 0 },
      mFreeVoices(), mActiveVoices(), mActiveVoiceData(0, c.allocator()), //todo - need allocator for queue/deque?
      mFreqs(get<kNVoices>().max(), c.allocator()),
      mLogMags(get<kNVoices>().max(), c.allocator()),
      mVoiceIDs(get<kNVoices>().max(), c.allocator())
  {
    controlChannelsIn(2);
    controlChannelsOut({3, get<kNVoices>(), get<kNVoices>().max()});
    setInputLabels({"frequencies", "magnitudes"});
    setOutputLabels({"frequencies", "magnitudes", "voice IDs"});
    mTracking.init();
  }

  void init(index nVoices)
  {
      if (!mActiveVoices.empty()) { mActiveVoices.pop_back(); }
      if (!mFreeVoices.empty()) { mFreeVoices.pop(); }
      for (index i = 0; i < nVoices; ++i) { mFreeVoices.push(i); }
      mActiveVoiceData.resize(nVoices);
      for (VoicePeak each : mActiveVoiceData) { each = { 0, 0, 0 }; }
  }

  template <typename T>
  void process(std::vector<HostVector<T>>& input,
               std::vector<HostVector<T>>& output, FluidContext& c)
  {
    index nVoices = get<kNVoices>();

    bool inputSizeChanged = mInputSize != input[0].size();
    bool sizeParamChanged = mSizeTracker.changed(nVoices);

    if (inputSizeChanged || sizeParamChanged)
    {
      mInputSize = input[0].size();
      controlChannelsOut({3, nVoices}); //update the dynamic out size
      init(nVoices);
    }

    index lowerSize;
    bool unfilledVoices = false;
    if (input[0].size() >= nVoices)
    {
        lowerSize = nVoices;
    }
    else
    {
        lowerSize = input[0].size();
        unfilledVoices = true;
    }

    //mOut1(Slice(0, lowerSize)) <<= input[1](Slice(0, lowerSize));
    //mOut0(Slice(0, lowerSize)) <<= input[0](Slice(0, lowerSize));

    vector<SinePeak> incomingVoices(0, c.allocator());
    for (index i = 0; i < lowerSize; ++i)
    {
        if (input[1].row(i) != 0 && input[0].row(i) != 0)
        {
            incomingVoices.push_back({ input[0].row(i), input[1].row(i), false });
        }
    }
    
    if (true) //change this to IF INPUT = TYPE MAGNITUDE, if dB skip
    {
        for (SinePeak voice : incomingVoices)
        {
            voice.logMag = 20 * log10(std::max(voice.logMag, algorithm::epsilon));
        }
    }

    double maxAmp = -144;
    for (SinePeak voice : incomingVoices)
    {
        if (voice.logMag > maxAmp) { maxAmp = voice.logMag; }
    }

    mTracking.processFrame(incomingVoices, maxAmp, get<kMinTrackLen>(), get<kBirthLowThreshold>(), get<kBirthHighTreshold>(), get<kTrackMethod>(), get<kTrackMagRange>(), get<kTrackFreqRange>(), get<kTrackProb>(), c.allocator());

    vector<VoicePeak> outgoingVoices(0, c.allocator());
    outgoingVoices = allocatorAlgorithm(mTracking.getActiveVoices(c.allocator()), nVoices, c.allocator());

    for (index i = 0; i < nVoices; ++i)
    {
        output[2].row(i) = outgoingVoices[i].voiceID;
        output[1].row(i) = outgoingVoices[i].logMag;
        output[0].row(i) = outgoingVoices[i].freq;
    }

    //output[2](Slice(0, lowerSize)) <<= mVoiceIDs(Slice(0, lowerSize));
    //output[1](Slice(0, lowerSize)) <<= mLogMags(Slice(0, lowerSize));
    //output[0](Slice(0, lowerSize)) <<= mFreqs(Slice(0, lowerSize));
    //
    //if (unfilledVoices)
    //{
    //    index unfilledVoicesLength = nVoices - lowerSize;
    //    output[2](Slice(lowerSize, unfilledVoicesLength)).fill(-1);
    //    output[1](Slice(lowerSize, unfilledVoicesLength)).fill(0);
    //    output[0](Slice(lowerSize, unfilledVoicesLength)).fill(0);
    //}
  }

  vector<VoicePeak> allocatorAlgorithm(vector<VoicePeak>& incomingVoices, index nVoices, Allocator& alloc)
  {
      //handle existing voices - killing or sustaining
      for (index existing = 0; existing < mActiveVoices.size(); ++existing)
      {
          bool killVoice = true;
          for (index incoming = 0; incoming < incomingVoices.size(); ++incoming)
          {
              //remove incoming voice events & allows corresponding voice to live if it already exists
              if (mActiveVoiceData[mActiveVoices[existing]].voiceID == incomingVoices[incoming].voiceID)
              {
                  killVoice = false;
                  incomingVoices.erase(incomingVoices.begin() + incoming);
                  break;
              }
          }
          if (killVoice) //note off
          {
              mActiveVoiceData[mActiveVoices[existing]] = { 0, 0, 0 };
              mFreeVoices.push(mActiveVoices[existing]);
              mActiveVoices.erase(mActiveVoices.begin() + existing);
          }
      }

      //handle new voice allocation
      for (index incoming = 0; incoming < incomingVoices.size(); ++incoming)
      {
          if (!mFreeVoices.empty())
          {
              index newVoiceIndex = mFreeVoices.front();
              mFreeVoices.pop();
              mActiveVoices.push_back(newVoiceIndex);
              mActiveVoiceData[newVoiceIndex] = incomingVoices[incoming];
          }
          else //voice stealing
          {
              ;
          }
      }

      return mActiveVoiceData;
  }

  MessageResult<void> clear()
  {
    //    mAlgorithm.init(get<0>(),mInputSize);
    return {};
  }

  static auto getMessageDescriptors()
  {
    return defineMessages(makeMessage("clear", &VoiceAllocatorClient::clear));
  }

  index latency() const { return 0; }

private:
  //  algorithm::RunningStats mAlgorithm;
    std::queue<index>                           mFreeVoices;
    std::deque<index>                           mActiveVoices;
    vector<VoicePeak>                           mActiveVoiceData;
    algorithm::PartialTracking                  mTracking;
    index                                       mInputSize;
    ParameterTrackChanges<index>                mSizeTracker;
    FluidTensor<double, 1>                      mFreqs;
    FluidTensor<double, 1>                      mLogMags;
    FluidTensor<double, 1>                      mVoiceIDs;
};

} // namespace voiceallocator

using VoiceAllocatorClient =
    ClientWrapper<voiceallocator::VoiceAllocatorClient>;

} // namespace client
} // namespace fluid
