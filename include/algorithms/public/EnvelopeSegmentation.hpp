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

class EnvelopeSegmentation
{

  using ArrayXd = Eigen::ArrayXd;

public:
  void init(double hiPassFreq, int fastRampUpTime, int slowRampUpTime,
            int fastRampDownTime, int slowRampDownTime, double onThreshold,
            double offThreshold, double floor, int debounce)
  {
    mHiPassFreq = hiPassFreq;
    mFastRampUpTime = fastRampUpTime;
    mSlowRampUpTime = slowRampUpTime;
    mFastRampDownTime = fastRampDownTime;
    mSlowRampDownTime = slowRampDownTime;
    mOnThreshold = onThreshold;
    mOffThreshold = offThreshold;
    mFloor = floor;
    mDebounce = debounce;
    initFilters();
    initSlides();
    mInitialized = true;
  }

  void updateParams(double hiPassFreq, int fastRampUpTime, int slowRampUpTime,
                    int fastRampDownTime, int slowRampDownTime,
                    double onThreshold, double offThreshold, double floor,
                    int debounce)
  {
    if (hiPassFreq != mHiPassFreq)
    {
      mHiPassFreq = hiPassFreq;
      initFilters();
    }

    if (fastRampUpTime != mFastRampUpTime ||
        fastRampDownTime != mFastRampDownTime)
    {
      mFastRampUpTime = fastRampUpTime;
      mFastRampDownTime = fastRampDownTime;
      mFastSlide.init(mFastRampUpTime, mFastRampDownTime, mFloor);
    }

    if (slowRampUpTime != mSlowRampUpTime ||
        slowRampDownTime != mSlowRampDownTime)
    {
      mSlowRampUpTime = slowRampUpTime;
      mSlowRampDownTime = slowRampDownTime;
      mSlowSlide.init(mSlowRampUpTime, mSlowRampDownTime, mFloor);
    }
    mOnThreshold = onThreshold;
    mOffThreshold = offThreshold;
    mFloor = floor;
    mDebounce = debounce;
  }

  bool initialized() { return mInitialized; }

  double processSample(const double in)
  {
    assert(mInitialized);
    double filtered = in;
    if (mHiPassFreq > 0)
      filtered = mHiPass2.processSample(mHiPass1.processSample(in));
    double rectified = std::abs(filtered);
    double dB = 20 * std::log10(rectified);
    double clipped = std::max(dB, mFloor);
    double fast = mFastSlide.processSample(clipped);
    double slow = mSlowSlide.processSample(clipped);
    double value = fast - slow;
    double detected = 0;

    if (!mState && value > mOnThreshold && mPrevValue < mOnThreshold &&
        mDebounceCount == 0)
    {
      detected = 1.0;
      mDebounceCount = mDebounce;
      mState = true;
    }
    else
    {
      if (mDebounceCount > 0) mDebounceCount--;
    }

    if (mState && value < mOffThreshold) { mState = false; }

    mPrevValue = value;
    return detected;
  }

private:
  void initFilters()
  {
    mHiPass1.init(mHiPassFreq);
    mHiPass2.init(mHiPassFreq);
  }

  void initSlides()
  {
    mFastSlide.init(mFastRampUpTime, mFastRampDownTime, mFloor);
    mSlowSlide.init(mSlowRampUpTime, mSlowRampDownTime, mFloor);
  }

  double              mHiPassFreq{0.2};
  int                 mFastRampUpTime{100};
  int                 mFastRampDownTime{100};
  int                 mSlowRampUpTime{100};
  int                 mSlowRampDownTime{100};
  double              mOnThreshold{-33};
  double              mOffThreshold{-42};
  ButterworthHPFilter mHiPass1;
  ButterworthHPFilter mHiPass2;
  SlideUDFilter       mFastSlide;
  SlideUDFilter       mSlowSlide;
  bool                mInitialized{false};
  double              mFloor{-45};
  int                 mDebounce{2};
  int                 mDebounceCount{1};
  double              mPrevValue;
  bool                mState{false};
};
}; // namespace algorithm
}; // namespace fluid
