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
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>
#include <cmath>

namespace fluid {
namespace algorithm {

class Envelope
{
public:
  void init(double floor, double hiPassFreq)
  {
    mFastSlide.init(floor);
    mSlowSlide.init(floor);
    initFilters(hiPassFreq);
    mHiPassFreq = hiPassFreq;
    mInitialized = true;
  }

  double processSample(const double in, double floor, index fastRampUpTime,
                       index slowRampUpTime, index fastRampDownTime,
                       index slowRampDownTime, double hiPassFreq)
  {
    using namespace std;
    assert(mInitialized);
    mFastSlide.updateCoeffs(fastRampUpTime, fastRampDownTime);
    mSlowSlide.updateCoeffs(slowRampUpTime, slowRampDownTime);
    if (std::isfinite(in)) mPrevValid = in;
    double filtered = mPrevValid;
    if (hiPassFreq != mHiPassFreq)
    {
      initFilters(hiPassFreq);
      mHiPassFreq = hiPassFreq;
    }
    if (mHiPassFreq > 0)
    {
      filtered = mHiPass2.processSample(mHiPass1.processSample(filtered));
    }
    double rectified = abs(filtered);
    double dB = 20 * log10(rectified);
    double clipped = max(dB, floor);
    double fast = mFastSlide.processSample(clipped);
    double slow = mSlowSlide.processSample(clipped);
    return fast - slow;
  }

  bool initialized() const { return mInitialized; }

private:
  void initFilters(double cutoff)
  {
    mHiPass1.init(cutoff);
    mHiPass2.init(cutoff);
  }

  double mHiPassFreq{0};
  bool   mInitialized{false};
  double mPrevValid{0};

  ButterworthHPFilter mHiPass1;
  ButterworthHPFilter mHiPass2;
  SlideUDFilter       mFastSlide;
  SlideUDFilter       mSlowSlide;
};
} // namespace algorithm
} // namespace fluid
