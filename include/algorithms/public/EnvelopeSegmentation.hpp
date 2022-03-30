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

#include "Envelope.hpp"
#include "../util/ButterworthHPFilter.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../util/SlideUDFilter.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>
#include <cmath>

namespace fluid {
namespace algorithm {

class EnvelopeSegmentation
{

  using ArrayXd = Eigen::ArrayXd;

public:
  void init(double floor, double hiPassFreq)
  {
    // mFastSlide.init(floor);
    // mSlowSlide.init(floor);
    mEnvelope.init(floor,hiPassFreq); 
    mDebounceCount = 1;
    // initFilters(hiPassFreq);
    // mHiPassFreq = hiPassFreq;
    mPrevValue = 0;
    mState = false;
    // mInitialized = true;
  }

  double processSample(const double in, double onThreshold, double offThreshold,
                       double floor, index fastRampUpTime, index slowRampUpTime,
                       index fastRampDownTime, index slowRampDownTime,
                       double hiPassFreq, index debounce)
  {
    // using namespace std;
    // assert(mInitialized);
    // mFastSlide.updateCoeffs(fastRampUpTime, fastRampDownTime);
    // mSlowSlide.updateCoeffs(slowRampUpTime, slowRampDownTime);
    // double filtered = in;
    // if (hiPassFreq != mHiPassFreq)
    // {
    //   initFilters(hiPassFreq);
    //   mHiPassFreq = hiPassFreq;
    // }
    // if (mHiPassFreq > 0){
    //   filtered = mHiPass2.processSample(mHiPass1.processSample(in));
    // }
    // double rectified = abs(filtered);
    // double dB = 20 * log10(rectified);
    // double clipped = max(dB, floor);
    // double fast = mFastSlide.processSample(clipped);
    // double slow = mSlowSlide.processSample(clipped);
    // double value = fast - slow;


    double value =
        mEnvelope.processSample(in, floor, fastRampUpTime, slowRampUpTime,
                                fastRampDownTime, slowRampUpTime, hiPassFreq);
    double detected = 0;

    if (!mState && value > onThreshold && mPrevValue < onThreshold &&
        mDebounceCount == 0)
    {
      detected = 1.0;
      mDebounceCount = debounce;
      mState = true;
    }
    else
    {
      if (mDebounceCount > 0) mDebounceCount--;
    }
    if (mState && value < offThreshold) { mState = false; }
    mPrevValue = value;
    return detected;
  }

  bool initialized() { return mEnvelope.initialized(); }

private:
//  void initFilters(double cutoff)
//  {
//    mHiPass1.init(cutoff);
//    mHiPass2.init(cutoff);
//  }
  
  Envelope mEnvelope; 
  // double mHiPassFreq{0};
  index  mDebounceCount{1};
  double mPrevValue{0};
  // bool   mInitialized{false};
  bool   mState{false};

  // ButterworthHPFilter mHiPass1;
  // ButterworthHPFilter mHiPass2;
  // SlideUDFilter       mFastSlide;
  // SlideUDFilter       mSlowSlide;
};
} // namespace algorithm
} // namespace fluid
