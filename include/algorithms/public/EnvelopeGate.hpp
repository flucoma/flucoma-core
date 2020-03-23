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

#include "../util/ButterworthHPFilter.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../util/SlideUDFilter.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>
#include <algorithm>
#include <cmath>

namespace fluid {
namespace algorithm {

class EnvelopeGate
{

  using ArrayXd = Eigen::ArrayXd;

public:
  EnvelopeGate(index maxSize)
  {
    mInputStorage = ArrayXd(maxSize);
    mOutputStorage = ArrayXd(maxSize);
  }

  void init(double hiPassFreq, index rampUpTime, index rampDownTime,
            double onThreshold, index mindeximeAboveThreshold,
            index minEventDuration, index upwardLookupTime, double offThreshold,
            index mindeximeBelowThreshold, index minSilenceDuration,
            index downwardLookupTime)
  {
    mHiPassFreq = hiPassFreq;
    mRampUpTime = rampUpTime;
    mRampDownTime = rampDownTime;
    mOnThreshold = onThreshold;
    mMindeximeAboveThreshold = mindeximeAboveThreshold;
    mMinEventDuration = minEventDuration;
    mUpwardLookupTime = upwardLookupTime;
    mOffThreshold = offThreshold;
    mFloor = std::min(mOffThreshold, mOnThreshold) - 1;
    mMindeximeBelowThreshold = mindeximeBelowThreshold,
    mMinSilenceDuration = minSilenceDuration;
    mDownwardLookupTime = downwardLookupTime;
    mDownwardLatency =
        std::max<index>(mindeximeBelowThreshold, mDownwardLookupTime);
    mLatency = std::max<index>(mMindeximeAboveThreshold + mUpwardLookupTime,
                               mDownwardLatency);
    if (mLatency < 0) mLatency = 1;
    initBuffers();
    initFilters();
    initSlide();
    mInitialized = true;
  }

  void updateParams(double hiPassFreq, index rampUpTime, index rampDownTime,
                    double onThreshold, index minEventDuration,
                    double offThreshold, index minSilenceDuration)
  {
    if (mHiPassFreq != hiPassFreq)
    {
      mHiPassFreq = hiPassFreq;
      initFilters();
    }
    if (mRampUpTime != rampUpTime || mRampDownTime != rampDownTime)
    {
      mRampUpTime = rampUpTime;
      mRampDownTime = rampDownTime;
      mSlide.updateCoeffs(mRampUpTime, mRampDownTime);
    }
    mOnThreshold = onThreshold;
    mMinEventDuration = minEventDuration;
    mOffThreshold = offThreshold;
    mMinSilenceDuration = minSilenceDuration;
    mFloor = std::min(mOffThreshold, mOnThreshold)  - 1;
  }

  index getLatency() { return mLatency; }
  bool  initialized() { return mInitialized; }

  double processSample(const double in)
  {
    assert(mInitialized);
    double filtered = in;
    if (mHiPassFreq > 0)
      filtered = mHiPass2.processSample(mHiPass1.processSample(in));
    double rectified = std::abs(filtered);
    double dB = 20 * std::log10(rectified);
    double clipped = std::max(dB, mFloor);
    double smoothed = mSlide.processSample(clipped);
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
      if (!mOutputState && mOnStateCount >= mMindeximeAboveThreshold &&
          mFillCount >= mLatency)
      {
        index onsetIndex =
            refineStart(mLatency - mMindeximeAboveThreshold - mUpwardLookupTime,
                        mUpwardLookupTime);
        mOutputBuffer.segment(onsetIndex, mLatency - onsetIndex) = 1;
        mEventCount = mOnStateCount;
        mOutputState = true; // we are officially on
      }
      else if (mOutputState && mOffStateCount >= mDownwardLatency &&
               mFillCount >= mLatency)
      {

        index offsetIndex =
            refineStart(mLatency - mDownwardLatency, mDownwardLookupTime);
        mOutputBuffer.segment(offsetIndex, mLatency - offsetIndex) = 0;
        mSilenceCount = mOffStateCount;
        mOutputState = false; // we are officially off
      }

      mOutputBuffer(mLatency - 1) = mOutputState ? 1 : 0;
      mInputState = nextState;
    }
    if (mLatency > 1)
    {
      mOutputBuffer.segment(0, mLatency - 1) =
          mOutputBuffer.segment(1, mLatency - 1);

      mInputBuffer.segment(0, mLatency - 1) =
          mInputBuffer.segment(1, mLatency - 1);
    }
    mInputBuffer(mLatency - 1) = smoothed;
    if (mFillCount < mLatency) mFillCount++;
    return mOutputBuffer(0);
  }

private:
  void initBuffers()
  {
    mInputBuffer = mInputStorage.segment(0, std::max<index>(mLatency, 1))
                       .setConstant(mFloor);
    mOutputBuffer =
        mOutputStorage.segment(0, std::max<index>(mLatency, 1)).setZero();
    mInputState = false;
    mOutputState = false;
    mFillCount = std::max<index>(mLatency, 1);
  }

  void initFilters()
  {
    mHiPass1.init(mHiPassFreq);
    mHiPass2.init(mHiPassFreq);
  }

  void initSlide() { mSlide.init(mRampUpTime, mRampDownTime, mFloor); }

  index refineStart(index start, index nSamples)
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
    {
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

  index               mLatency;
  index               mFillCount;
  double              mHiPassFreq{0.2};
  index               mRampUpTime{100};
  index               mRampDownTime{100};
  double              mOnThreshold{-33};
  index               mMindeximeAboveThreshold{440};
  index               mMinEventDuration{440};
  index               mDownwardLookupTime{10};
  index               mDownwardLatency;
  double              mOffThreshold{-42};
  index               mMindeximeBelowThreshold{10};
  index               mMinSilenceDuration{10};
  index               mUpwardLookupTime{24};
  ArrayXd             mInputBuffer;
  ArrayXd             mOutputBuffer;
  ArrayXd             mInputStorage;
  ArrayXd             mOutputStorage;
  bool                mInputState{false};
  bool                mOutputState{false};
  ButterworthHPFilter mHiPass1;
  ButterworthHPFilter mHiPass2;
  SlideUDFilter       mSlide;
  index               mOnStateCount{0};
  index               mOffStateCount{0};
  index               mEventCount{0};
  index               mSilenceCount{0};
  bool                mInitialized{false};
  double              mFloor{-45};
};
} // namespace algorithm
} // namespace fluid
