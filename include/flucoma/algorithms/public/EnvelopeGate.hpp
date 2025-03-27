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

#include "../util/ButterworthHPFilter.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../util/SlideUDFilter.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/FluidMemory.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>
#include <cmath>

namespace fluid {
namespace algorithm {

class EnvelopeGate
{

  using ArrayXd = Eigen::ArrayXd;

public:
  EnvelopeGate(index maxSize, Allocator& alloc = FluidDefaultAllocator())
      : mInputBuffer(maxSize, alloc), mOutputBuffer(maxSize, alloc)
  {}

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
    assert(mLatency <= mInputBuffer.size());
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

    if (std::isfinite(in)) mPrevValid = in;
    double filtered = mPrevValid;
    if (hiPassFreq != mHiPassFreq)
    {
      initFilters(hiPassFreq);
      mHiPassFreq = hiPassFreq;
    }
    if (mHiPassFreq > 0)
      filtered = mHiPass2.processSample(mHiPass1.processSample(filtered));

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
        mOutputBuffer(mWriteHead) = 1;
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
        mOutputBuffer(mWriteHead) = 0;
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
            refineStart(mWriteHead - mMinTimeAboveThreshold - mUpwardLookupTime,
                        mUpwardLookupTime);

        index blockSize = mWriteHead > onsetIndex
                              ? mWriteHead - onsetIndex
                              : (mLatency - onsetIndex) + mWriteHead;

        index size = onsetIndex + blockSize > mLatency ? mLatency - onsetIndex
                                                       : blockSize;

        mOutputBuffer.segment(onsetIndex, size) = 1;
        mOutputBuffer.segment(0, blockSize - size) = 1;

        mEventCount = mOnStateCount;
        mOutputState = true; // we are officially on
      }
      else if (mOutputState && mOffStateCount >= mDownwardLatency &&
               mFillCount >= mLatency)
      {

        index offsetIndex =
            refineStart(mWriteHead - mDownwardLatency, mDownwardLookupTime);

        index blockSize = mWriteHead > offsetIndex
                              ? mWriteHead - offsetIndex
                              : (mLatency - offsetIndex) + mWriteHead;

        index size = offsetIndex + blockSize > mLatency ? mLatency - offsetIndex
                                                        : blockSize;

        mOutputBuffer.segment(offsetIndex, size) = 0;
        mOutputBuffer.segment(0, blockSize - size) = 0;

        mSilenceCount = mOffStateCount;
        mOutputState = false; // we are officially off
      }

      mOutputBuffer(mWriteHead) = mOutputState ? 1 : 0;

      mInputState = nextState;
    }

    mInputBuffer(mWriteHead) = smoothed;

    if (mFillCount < mLatency) mFillCount++;
    double result = mOutputBuffer(mReadHead);

    if (++mWriteHead >= max<index>(mLatency, 1)) mWriteHead = 0;
    if (++mReadHead >= max<index>(mLatency, 1)) mReadHead = 0;

    return result;
  }
  index getLatency() const { return mLatency; }
  bool  initialized() const { return mInitialized; }


private:
  void initBuffers(double initialValue)
  {
    using namespace std;
    mInputBuffer.segment(0, max<index>(mLatency, 1)).setConstant(initialValue);
    mOutputBuffer.segment(0, max<index>(mLatency, 1)).setZero();
    mInputState = false;
    mOutputState = false;
    mFillCount = max<index>(mLatency, 1);
    mWriteHead = max<index>(mLatency, 1) - 1;
    mReadHead = 0;
  }

  void initFilters(double cutoff)
  {
    mHiPass1.init(cutoff);
    mHiPass2.init(cutoff);
  }

  index refineStart(index start, index nSamples)
  {

    using Eigen::Array2d;
    using Eigen::Array2i;

    index circularStart = start < 0 ? mLatency + start : start;

    if (nSamples < 2)
      return circularStart + nSamples < mLatency
                 ? circularStart + nSamples
                 : circularStart + nSamples - mLatency;

    index circularNSamples = circularStart + nSamples > mLatency
                                 ? mLatency - circularStart
                                 : nSamples;

    if (circularNSamples == nSamples)
    {
      index argMin;
      mInputBuffer.segment(circularStart, nSamples).minCoeff(&argMin);
      return circularStart + argMin;
    }
    else
    {
      Array2i argMins;
      mInputBuffer.segment(circularStart, circularNSamples)
          .minCoeff(argMins.data());
      mInputBuffer.segment(0, nSamples - circularNSamples)
          .minCoeff(argMins.data() + 1);
      Array2d mins = {
          mInputBuffer.segment(circularStart, circularNSamples)(argMins(0)),
          mInputBuffer.segment(0, nSamples - circularNSamples)(argMins(1))};
      index whichArgMin;
      mins.minCoeff(&whichArgMin);

      return whichArgMin == 0 ? circularStart + argMins(0) : argMins(1);
    }
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
    else if (mInputState && nextState) { mOnStateCount++; }
    else if (!mInputState && !nextState) { mOffStateCount++; }
  }

  index  mLatency;
  index  mFillCount;
  double mHiPassFreq{0};
  double mPrevValid{0};

  index mMinTimeAboveThreshold{440};
  index mDownwardLookupTime{10};
  index mDownwardLatency;
  index mMinTimeBelowThreshold{10};
  index mUpwardLookupTime{24};

  ScopedEigenMap<ArrayXd> mInputBuffer;
  ScopedEigenMap<ArrayXd> mOutputBuffer;

  index mWriteHead;
  index mReadHead;

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
