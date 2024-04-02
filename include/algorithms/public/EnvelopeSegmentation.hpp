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
public:
  void init(double floor, double hiPassFreq)
  {
    mEnvelope.init(floor,hiPassFreq);
    mDebounceCount = 0;
    mPrevValue = 0;
    mState = false;
  }

  double processSample(const double in, double onThreshold, double offThreshold,
                       double floor, index fastRampUpTime, index slowRampUpTime,
                       index fastRampDownTime, index slowRampDownTime,
                       double hiPassFreq, index debounce)
  {
    double value =
        mEnvelope.processSample(in, floor, fastRampUpTime, slowRampUpTime,
                                fastRampDownTime, slowRampDownTime, hiPassFreq);
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

  bool initialized() const { return mEnvelope.initialized(); }

private:
  Envelope mEnvelope;
  index  mDebounceCount{0};
  double mPrevValue{0};
  bool   mState{false};
};
} // namespace algorithm
} // namespace fluid
