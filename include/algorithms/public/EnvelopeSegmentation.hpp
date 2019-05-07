#pragma once

#include "../../data/TensorTypes.hpp"
#include "../util/ButterworthHPFilter.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../util/SlideUDFilter.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>

namespace fluid {
namespace algorithm {

using _impl::asEigen;
using _impl::asFluid;
using Eigen::Array;
using Eigen::ArrayXd;

class EnvelopeSegmentation {

public:
  EnvelopeSegmentation(size_t maxSize, int outputType)
      : mMaxSize(maxSize), mHiPassFreq(0.2), mRampUpTime(100),
        mRampDownTime(100), mOnThreshold(-33), mMinTimeAboveThreshold(440),
        mMinEventDuration(440), mUpwardLookupTime(24), mOffThreshold(-42),
        mMinTimeBelowThreshold(10), mMinSilenceDuration(10),
        mDownwardLookupTime(10), mOutputType(outputType), mOnStateCount(0),
        mOffStateCount(0), mEventCount(0), mSilenceCount(0) {
    mInputStorage = ArrayXd(maxSize);
    mOutputStorage = ArrayXd(maxSize);
    initBuffers();
    initFilters();
  }
  void init(double hiPassFreq, int rampUpTime, int rampDownTime,
            double onThreshold, int minTimeAboveThreshold, int minEventDuration,
            int upwardLookupTime, double offThreshold,
            int minTimeBelowThreshold, int minSilenceDuration,
            int downwardLookupTime) {
    mHiPassFreq = hiPassFreq;
    mRampUpTime = rampUpTime;
    mRampDownTime = rampDownTime;
    mOnThreshold = onThreshold;
    mMinTimeAboveThreshold = minTimeAboveThreshold;
    mMinEventDuration = minEventDuration;
    mUpwardLookupTime = upwardLookupTime;
    mOffThreshold = offThreshold;
    mMinTimeBelowThreshold = minTimeBelowThreshold,
    mMinSilenceDuration = minSilenceDuration;
    mDownwardLookupTime = downwardLookupTime;
    mLatency = std::max(mMinTimeAboveThreshold + mUpwardLookupTime,
                        mMinTimeBelowThreshold + mDownwardLookupTime);
    if (mLatency == 0)
      mLatency = 1;
    std::cout << "latency " << mLatency << std::endl;
    assert(mLatency <= mMaxSize);
    initBuffers();
    initFilters();
  }
  void initBuffers() {
    mInputBuffer = mInputStorage.segment(0, std::max(mLatency, 1));
    mOutputBuffer = mInputStorage.segment(0, std::max(mLatency, 1));
    mInputState = false;
    mOutputState = false;
    mFillCount = 0;
  }

  void initFilters() {
    mHiPass1.init(mHiPassFreq);
    mHiPass2.init(mHiPassFreq);
    mSlide.init(mRampUpTime, mRampDownTime);
  }

  int refineStart(int nSamples) {
    if (nSamples < 2)
      return nSamples;
    ArrayXd diff = mInputBuffer.segment(1, nSamples - 1) -
                   mInputBuffer.segment(0, nSamples - 1);
    ArrayXd::Index index;
    diff.maxCoeff(&index);
    return index;
  }

  void updateCounters(bool nextState) {
    if (!mInputState && nextState) { // change from 0 to 1
      mOffStateCount = 0;
    } else if (mInputState && !nextState) {
      mOnStateCount = 0;
    } else if (mInputState && nextState) {
      mOnStateCount++;
    } else if (!mInputState && !nextState) {
      mOffStateCount++;
    }
  }

  void processSample(const RealVectorView input,
                     RealVectorView output) { // size is supposed to be 1
    double in = input(0);
    double filtered = mHiPass2.processSample(mHiPass1.processSample(in));
    double rectified =
        std::max(std::abs(filtered), 6.3095734448019e-08); // -144dB
    double dB = 20 * std::log10(rectified);
    double smoothed = mSlide.processSample(dB);
    bool forcedState = false;
    // case 1: we are waiting for event to finish
    if (mOutputState && mEventCount > 0) {
      if (mEventCount >= mMinEventDuration) {
        mEventCount = 0;
      } else {
        forcedState = true;
        mOutputBuffer(mLatency - 1) = 1;
        mEventCount++;
      }
      // case 2: we are waiting for silence to finish
    } else if (!mOutputState && mSilenceCount > 0) {
      if (mSilenceCount >= mMinSilenceDuration) {
        mSilenceCount = 0;
      } else {
        forcedState = true;
        mOutputBuffer(mLatency - 1) = 0;
        mSilenceCount++;
      }
    }
    // case 3: need to compute state
    if (!forcedState) {
      bool nextState = mInputState;
      if (!mInputState && smoothed > mOnThreshold)
        nextState = true;
      if (mInputState && smoothed < mOffThreshold)
        nextState = false;
      updateCounters(nextState);
      mOutputBuffer(mLatency - 1) = mOutputState ? 1 : 0;
      // establish and refine
      if (!mOutputState && mOnStateCount > mMinTimeAboveThreshold &&
          mFillCount >= mLatency) {
        int onsetIndex = refineStart(mUpwardLookupTime);
        mOutputBuffer.segment(onsetIndex, mLatency - onsetIndex) = 1;
        mEventCount = mOnStateCount;
        mOutputState = true; // we are officially on
      } else if (mOutputState && mOffStateCount > mMinTimeBelowThreshold &&
                 mFillCount >= mLatency) {
        int offsetIndex = refineStart(mDownwardLookupTime);
        mOutputBuffer.segment(offsetIndex, mLatency - offsetIndex) = 0;
        mSilenceCount = mOffStateCount;
        mOutputState = false; // we are officially off
      }
      mInputState = nextState;
    }
    switch (mOutputType) {
    case 0:
      output(0) = mOutputBuffer(0);
      break;
    case 1:
      output(0) = filtered;
      break;
    case 2:
      output(0) = std::pow(10, smoothed / 20);
      break;
    case 3:
      output(0) =  mInputBuffer(1) - mInputBuffer(0);
    }
    if (mLatency > 1) {
      mOutputBuffer.segment(0, mLatency - 1) =
          mOutputBuffer.segment(1, mLatency - 1);

      mInputBuffer.segment(0, mLatency - 1) =
          mInputBuffer.segment(1, mLatency - 1);
    }
    mInputBuffer(mLatency - 1) = smoothed;
    if (mFillCount < mLatency)
      mFillCount++;
  }

private:
  int mMaxSize;
  int mLatency;
  int mFillCount;
  double mHiPassFreq;
  int mRampUpTime;
  int mRampDownTime;
  double mOnThreshold;
  int mMinTimeAboveThreshold;
  int mMinEventDuration;
  int mDownwardLookupTime;
  double mOffThreshold;
  int mMinTimeBelowThreshold;
  int mMinSilenceDuration;
  int mUpwardLookupTime;
  int mOutputType;
  ArrayXd mInputBuffer;
  ArrayXd mOutputBuffer;
  ArrayXd mInputStorage;
  ArrayXd mOutputStorage;
  bool mInputState;
  bool mOutputState;
  ButterworthHPFilter mHiPass1;
  ButterworthHPFilter mHiPass2;
  SlideUDFilter mSlide;
  int mOnStateCount;
  int mOffStateCount;
  int mEventCount;
  int mSilenceCount;
};
}; // namespace algorithm
}; // namespace fluid
