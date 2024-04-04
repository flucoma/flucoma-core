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

#include "../util/PartialTracking.hpp"

namespace fluid {
namespace algorithm {

class VoiceAllocator
{
  template <typename T>
  using vector = rt::vector<T>;

public:
  VoiceAllocator(index nVoices, Allocator& alloc)
    : mTracking(alloc), mVoices{ nVoices },
    mFreeVoices(alloc), mActiveVoices(alloc),
    mActiveVoiceData(0, alloc)
  {}

  void init(index nVoices, Allocator& alloc)
  {
    mVoices = nVoices;
    while (!mActiveVoices.empty()) { mActiveVoices.pop_back(); }
    while (!mFreeVoices.empty()) { mFreeVoices.pop(); }
    for (index i = 0; i < nVoices; ++i) { mFreeVoices.push(i); }
    mActiveVoiceData.resize(nVoices);
    for (VoicePeak each : mActiveVoiceData) { each = { 0, 0, 0 }; }
    mTracking.init();
    mInitialized = true;
  }

  void processFrame(vector<SinePeak> incomingVoices, vector<VoicePeak>& outgoingVoices, index minTrackLen, double birthLowTreshold, double birthHighTreshold, index trackMethod, double trackMagRange, double trackFreqRange, double trackProb, index sortMethod, Allocator& alloc)
  {
    assert(mInitialized);

    double maxAmp = -144;
    for (const SinePeak& voice : incomingVoices)
    {
      if (voice.logMag > maxAmp) { maxAmp = voice.logMag; }
    }

    mTracking.processFrame(incomingVoices, maxAmp, minTrackLen, birthLowTreshold, birthHighTreshold, trackMethod, trackMagRange, trackFreqRange, trackProb, alloc);

    outgoingVoices = mTracking.getActiveVoices(alloc);
    outgoingVoices = sortVoices(outgoingVoices, sortMethod);
    if (outgoingVoices.size() > mVoices)
      outgoingVoices.resize(mVoices);
    outgoingVoices = assignVoices(outgoingVoices, alloc);

    mTracking.prune();
  }

  void reset() {mInitialized = false;}

  bool initialized() const { return mInitialized; }

private:

  vector<VoicePeak> sortVoices(vector<VoicePeak>& incomingVoices, index sortingMethod)
  {
    switch (sortingMethod)
    {
    case 0: //lowest
      std::sort(incomingVoices.begin(), incomingVoices.end(),
                [](const VoicePeak& voice1, const VoicePeak& voice2)
                { return voice1.freq < voice2.freq; });
      break;
    case 1: //loudest
      std::sort(incomingVoices.begin(), incomingVoices.end(),
                [](const VoicePeak& voice1, const VoicePeak& voice2)
                { return voice1.logMag > voice2.logMag; });
      break;
    }
    return incomingVoices;
  }

  vector<VoicePeak> assignVoices(vector<VoicePeak>& incomingVoices, Allocator& alloc)
  {
    //move released to free
    for (index existing = 0; existing < mActiveVoiceData.size(); ++existing)
    {
      if (mActiveVoiceData[existing].state == algorithm::VoiceState::kReleaseState)
        mActiveVoiceData[existing].state = algorithm::VoiceState::kFreeState;
    }

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
          mActiveVoiceData[mActiveVoices[existing]] = incomingVoices[incoming]; //update freq/mag
          mActiveVoiceData[mActiveVoices[existing]].state = algorithm::VoiceState::kSustainState;
          incomingVoices.erase(incomingVoices.begin() + incoming);
          break;
        }
      }
      if (killVoice) //voice off
      {
        mActiveVoiceData[mActiveVoices[existing]].state = algorithm::VoiceState::kReleaseState;
        mFreeVoices.push(mActiveVoices[existing]);
        mActiveVoices.erase(mActiveVoices.begin() + existing);
        --existing;
      }
    }

    //handle new voice allocation
    for (index incoming = 0; incoming < incomingVoices.size(); ++incoming)
    {
      if (!mFreeVoices.empty()) //voice on
      {
        index newVoiceIndex = mFreeVoices.front();
        mFreeVoices.pop();
        mActiveVoices.push_back(newVoiceIndex);
        algorithm::VoiceState prevState = mActiveVoiceData[newVoiceIndex].state;
        mActiveVoiceData[newVoiceIndex] = incomingVoices[incoming];
        if (prevState == algorithm::VoiceState::kReleaseState) //mark as stolen
          mActiveVoiceData[newVoiceIndex].state = algorithm::VoiceState::kStolenState;
      }
    }

    return mActiveVoiceData;
  }

  PartialTracking         mTracking;
  index                   mVoices;
  rt::queue<index>        mFreeVoices;
  rt::deque<index>        mActiveVoices;
  vector<VoicePeak>       mActiveVoiceData;

  bool                    mInitialized{ false };
};
} // namespace algorithm
} // namespace fluid
