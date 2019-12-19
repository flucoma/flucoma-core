/*
Copyright 2017-2019 University of Huddersfield.
Licensed under the BSD-3 License.
See LICENSE file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/

#pragma once

#include "../util/ButterworthHPFilter.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../util/SlideUDFilter.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>
#include <algorithm>
#include <cmath>

namespace fluid {
namespace algorithm {

class EnvelopeSegmentation
{

  using ArrayXd = Eigen::ArrayXd;

public:
  EnvelopeSegmentation(size_t maxSize, int outputType)
      : mMaxSize(maxSize), mOutputType(outputType)
  {
    mInputStorage = ArrayXd(maxSize);
    mOutputStorage = ArrayXd(maxSize);
  }

  void init(double hiPassFreq, int rampUpTime, int rampUpTime2,
            int rampDownTime, int rampDownTime2, double onThreshold,
            double relOnThreshold, double relOffThreshold,
            int minTimeAboveThreshold, int minEventDuration,
            int upwardLookupTime, double offThreshold,
            int minTimeBelowThreshold, int minSilenceDuration,
            int downwardLookupTime)
  {
    mHiPassFreq = hiPassFreq;
    mRampUpTime = rampUpTime;
    mRampUpTime2 = rampUpTime2;
    mRampDownTime = rampDownTime;
    mRampDownTime2 = rampDownTime2;
    mOnThreshold = onThreshold;
    mRelOnThreshold = relOnThreshold;
    mRelOffThreshold = relOffThreshold;
    mMinTimeAboveThreshold = minTimeAboveThreshold;
    mMinEventDuration = minEventDuration;
    mUpwardLookupTime = upwardLookupTime;
    mOffThreshold = offThreshold;
    mMinTimeBelowThreshold = minTimeBelowThreshold,
    mMinSilenceDuration = minSilenceDuration;
    mDownwardLookupTime = downwardLookupTime;
    mDownwardLatency = std::max(minTimeBelowThreshold, mDownwardLookupTime);
    mLatency =
        std::max(mMinTimeAboveThreshold + mUpwardLookupTime, mDownwardLatency);
    if (mLatency < 0) mLatency = 1;
    assert(mLatency <= mMaxSize);
    initBuffers();
    initFilters();
    initSlides();
    mInitialized = true;
  }

  void updateParams (double hiPassFreq, int rampUpTime, int rampUpTime2,
                     int rampDownTime, int rampDownTime2, double onThreshold,
                     double relOnThreshold, double relOffThreshold,
                       int minEventDuration,
                       double offThreshold,
                     int minSilenceDuration) {
      if (mHiPassFreq != hiPassFreq){
          mHiPassFreq = hiPassFreq;
          initFilters();
      }
      if (mRampUpTime != rampUpTime || mRampDownTime != rampDownTime) {
          mRampUpTime = rampUpTime;
          mRampDownTime = rampDownTime;
          mSlide.updateCoeffs(mRampUpTime, mRampDownTime);
      }
      if (mRampUpTime2 != rampUpTime2 || mRampDownTime2 != rampDownTime2) {
          mRampUpTime2 = rampUpTime2;
          mRampDownTime2 = rampDownTime2;
          mSlide2.updateCoeffs(mRampUpTime2, mRampDownTime2);
      }
      mOnThreshold = onThreshold;
      mRelOnThreshold = relOnThreshold;
      mRelOffThreshold = relOffThreshold;
      mMinEventDuration = minEventDuration;
      mOffThreshold = offThreshold;
      mMinSilenceDuration = minSilenceDuration;
  }

  int  getLatency() { return mLatency; }
  bool initialized() { return mInitialized; }

  double processSample(const double in)
  {
    assert(mInitialized);
    double filtered = mHiPass2.processSample(mHiPass1.processSample(in));
      
      double rectified = std::abs(filtered);
     double dB = 20 * std::log10(rectified);
     double floor = std::max(dB, (std::min(mOffThreshold, mOnThreshold) - 1.));//need to remove a dB or so, to gain the advantage of not starting from too low (not from absolute silence floor makes it more nervous) but allowing a bit of headroom (for the expon slides to go down faster) - maybe a dithered version would be better (TODO)
     double smoothed = mSlide.processSample(floor);
     double smoothed2 = mSlide2.processSample(floor);

    double relEnv = smoothed2 - smoothed;
    bool   forcedState = false;
    // case 1: we are waiting for event to finish
    if (mOutputState && mEventCount > 0)
    {
      if (mEventCount >= mMinEventDuration) { mEventCount = 0; }
      else
      {
        forcedState = true;
        mOutputBuffer(mLatency - 1) = 1;
        mEventCount++;
      }
      // case 2: we are waiting for silence to finish
    }
    else if (!mOutputState && mSilenceCount > 0)
    {
      if (mSilenceCount >= mMinSilenceDuration) { mSilenceCount = 0; }
      else
      {
        forcedState = true;
        mOutputBuffer(mLatency - 1) = 0;
        mSilenceCount++;
      }
    }
    // case 3: need to compute state
    if (!forcedState)
    {
      bool nextState = mInputState;
      if (!mInputState && smoothed >= mOnThreshold) { nextState = true; }
      if (mInputState && smoothed <= mOffThreshold) { nextState = false; }
      updateCounters(nextState);
      // establish and refine
      if (!mOutputState && mOnStateCount >= mMinTimeAboveThreshold &&
          mFillCount >= mLatency)
      {
        int onsetIndex =
            refineStart(mLatency - mMinTimeAboveThreshold - mUpwardLookupTime,
                        mUpwardLookupTime, true);
        int numSamples = onsetIndex - mMinTimeAboveThreshold;
        mOutputBuffer.segment(onsetIndex, mLatency - onsetIndex) = 1;
        mEventCount = mOnStateCount;
        mOutputState = true; // we are officially on
      }
      else if (mOutputState && mOffStateCount >= mDownwardLatency &&
               mFillCount >= mLatency)
      {

        int offsetIndex = refineStart(mLatency - mDownwardLatency,
                                      mDownwardLookupTime, false);
        mOutputBuffer.segment(offsetIndex, mLatency - offsetIndex) = 0;
        mSilenceCount = mOffStateCount;
        mOutputState = false; // we are officially off
      }

      // enveloppe differential retrigging
      if (shouldRetrigger(relEnv))
      {
        mOutputBuffer(mLatency - 1) = 0;
        mEventCount = 1;
        mOutputState = true; // we are officially on, starting next sample
      }
      else
      {
        mOutputBuffer(mLatency - 1) = mOutputState ? 1 : 0;
      }
      if (relEnv < mRelOffThreshold && mRetriggerState) mRetriggerState = false;

      mInputState = nextState;
    }

////////////////////// to be removed and replace the return to case 0
    double output;
    switch (mOutputType)
    {
    case 0: output = mOutputBuffer(0); break;
    case 1: output = filtered; break;
    case 2: output = smoothed; break; //std::pow(10.0, smoothed / 20.0); break;
    case 3: output = relEnv; break; //std::pow(10.0, relEnv / 20.0); break;
    // case 4: output = mInputBuffer(1) - mInputBuffer(0);
    }
/////////////////////// up to here

    //
    if (mLatency > 1)
    {
      mOutputBuffer.segment(0, mLatency - 1) =
          mOutputBuffer.segment(1, mLatency - 1);

      mInputBuffer.segment(0, mLatency - 1) =
          mInputBuffer.segment(1, mLatency - 1);
    }
    mInputBuffer(mLatency - 1) = smoothed;
    if (mFillCount < mLatency) mFillCount++;
    return output;
  }

private:
  void initBuffers()
  {
      mInputBuffer = mInputStorage.segment(0, std::max(mLatency, 1)).setConstant(std::min(mOffThreshold, mOnThreshold) - 3.); //threshold of silence
    mOutputBuffer = mOutputStorage.segment(0, std::max(mLatency, 1)).setZero();
    mInputState = false;
    mOutputState = false;
    mRetriggerState = false;
    mFillCount = std::max(mLatency, 1); //has been off since the begining of time.
  }

  void initFilters()
  {
    mHiPass1.init(mHiPassFreq);
    mHiPass2.init(mHiPassFreq);
  }

  void initSlides() {
      mSlide.init(mRampUpTime, mRampDownTime, (std::min(mOffThreshold, mOnThreshold) - 3.));
      mSlide2.init(mRampUpTime2, mRampDownTime2, (std::min(mOffThreshold, mOnThreshold) - 3.));
}

  int refineStart(int start, int nSamples, bool direction = true)
  {
    if (nSamples < 2) return start + nSamples;
    ArrayXd        seg = mInputBuffer.segment(start, nSamples);
    ArrayXd::Index index;
    seg.minCoeff(&index);
    return start + index;
  }

  void updateCounters(bool nextState)
  {
    if (!mInputState && nextState)
    { // change from 0 to 1
      mOffStateCount = 0;
      mOnStateCount = 1;
    }
    else if (mInputState && !nextState)
    {
      mOnStateCount = 0;
      mOffStateCount = 1;
    }
    else if (mInputState && nextState)
    {
      mOnStateCount++;
    }
    else if (!mInputState && !nextState)
    {
      mOffStateCount++;
    }
  }

  bool shouldRetrigger(double relEnv)
  {
    if (mRetriggerState) return false;
    if (mInputState && mOutputState && mOnStateCount > 0 &&
        relEnv > mRelOnThreshold)
    {
      mRetriggerState = true;
      return true;
    }
    return false;
  }

  int                 mMaxSize;
  int                 mLatency;
  int                 mFillCount;
  double              mHiPassFreq{0.2};
  int                 mRampUpTime{100};
  int                 mRampUpTime2{50};
  int                 mRampDownTime{100};
  int                 mRampDownTime2{50};
  double              mOnThreshold{-33};
  double              mRelOnThreshold{-33};
  bool                mRetriggerState{false};
  int                 mMinTimeAboveThreshold{440};
  int                 mMinEventDuration{440};
  int                 mDownwardLookupTime{10};
  int                 mDownwardLatency;
  double              mOffThreshold{-42};
  double              mRelOffThreshold{-42};
  int                 mMinTimeBelowThreshold{10};
  int                 mMinSilenceDuration{10};
  int                 mUpwardLookupTime{24};
  int                 mOutputType;
  ArrayXd             mInputBuffer;
  ArrayXd             mOutputBuffer;
  ArrayXd             mInputStorage;
  ArrayXd             mOutputStorage;
  bool                mInputState{false};
  bool                mOutputState{false};
  ButterworthHPFilter mHiPass1;
  ButterworthHPFilter mHiPass2;
  SlideUDFilter       mSlide;
  SlideUDFilter       mSlide2;
  int                 mOnStateCount{0};
  int                 mOffStateCount{0};
  int                 mEventCount{0};
  int                 mSilenceCount{0};
  bool                mInitialized{false};
};
}; // namespace algorithm
}; // namespace fluid
