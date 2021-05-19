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
#include "../../data/FluidIndex.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>
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

  void init(double onThreshold, double offThreshold, double hiPassFreq,
            index minTimeAboveThreshold, index upwardLookupTime,
            index minTimeBelowThreshold, index downwardLookupTime)
  {
    using namespace std;

    mMinTimeAboveThreshold = minTimeAboveThreshold;
    mUpwardLookupTime = upwardLookupTime;
    mMinTimeBelowThreshold = minTimeBelowThreshold,
    mDownwardLookupTime = downwardLookupTime;
    mDownwardLatency = max<index>(minTimeBelowThreshold, mDownwardLookupTime);
    mLatency = max<index>(mMinTimeAboveThreshold + mUpwardLookupTime,
                          mDownwardLatency);
    if (mLatency < 0) mLatency = 1;
    mHiPassFreq = hiPassFreq;
    initFilters(mHiPassFreq);
    double initVal = min(onThreshold, offThreshold) - 1;
    initBuffers(initVal);
    mSlide.init(initVal);
    mInputState = false;
    mOutputState = false;
    mOnStateCount = 0;
    mOffStateCount = 0;
    mEventCount = 0;
    mSilenceCount = 0;
    mInitialized = true;
  }

  double processSample(const double in, double onThreshold, double offThreshold,
                       index rampUpTime, index rampDownTime, double hiPassFreq,
                       index minEventDuration, index minSilenceDuration)
  {
    using namespace std;
    assert(mInitialized);

    mSlide.updateCoeffs(rampUpTime, rampDownTime);

    double filtered = in;
    if (hiPassFreq != mHiPassFreq)
    {
      initFilters(hiPassFreq);
      mHiPassFreq = hiPassFreq;
    }
    if (mHiPassFreq > 0)
      filtered = mHiPass2.processSample(mHiPass1.processSample(in));

    double rectified = abs(filtered);
    double dB = 20 * log10(rectified);
    double floor = min(offThreshold, onThreshold) - 1;
    double clipped = max(dB, floor);
    double smoothed = mSlide.processSample(clipped);
    bool   forcedState = false;

    // case 1: we are waiting for event to finish
    if (mOutputState && mEventCount > 0)
    {
      if (mEventCount >= minEventDuration) { mEventCount = 0; }
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
      if (mSilenceCount >= minSilenceDuration) { mSilenceCount = 0; }
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
      if (!mInputState && smoothed >= onThreshold) { nextState = true; }
      if (mInputState && smoothed <= offThreshold) { nextState = false; }
      updateCounters(nextState);
      // establish and refine
      if (!mOutputState && mOnStateCount >= mMinTimeAboveThreshold &&
          mFillCount >= mLatency)
      {
        index onsetIndex =
            refineStart(mLatency - mMinTimeAboveThreshold - mUpwardLookupTime,
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

  index getLatency() { return mLatency; }
  bool  initialized() { return mInitialized; }


private:
  void initBuffers(double initialValue)
  {
    using namespace std;
    mInputBuffer = mInputStorage.segment(0, max<index>(mLatency, 1))
                       .setConstant(initialValue);
    mOutputBuffer =
        mOutputStorage.segment(0, max<index>(mLatency, 1)).setZero();
    mInputState = false;
    mOutputState = false;
    mFillCount = max<index>(mLatency, 1);
  }

  void initFilters(double cutoff)
  {
    mHiPass1.init(cutoff);
    mHiPass2.init(cutoff);
  }

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

  index  mLatency;
  index  mFillCount;
  double mHiPassFreq{0};

  index mMinTimeAboveThreshold{440};
  index mDownwardLookupTime{10};
  index mDownwardLatency;
  index mMinTimeBelowThreshold{10};
  index mUpwardLookupTime{24};

  ArrayXd mInputBuffer;
  ArrayXd mOutputBuffer;
  ArrayXd mInputStorage;
  ArrayXd mOutputStorage;

  bool mInputState{false};
  bool mOutputState{false};

  index mOnStateCount{0};
  index mOffStateCount{0};
  index mEventCount{0};
  index mSilenceCount{0};
  bool  mInitialized{false};

  ButterworthHPFilter mHiPass1;
  ButterworthHPFilter mHiPass2;
  SlideUDFilter       mSlide;
};
} // namespace algorithm
} // namespace fluid
