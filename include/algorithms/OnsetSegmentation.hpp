
#pragma once

#include "Descriptors.hpp"

namespace fluid {
namespace segmentation {
  
class OnsetSegmentation
{
  using Real = fluid::FluidTensorView<double, 1>;

public:

  enum Normalisation
  {
    kNone,
    kAmplitude,
    kPower,
  };
  
  enum DifferenceFunction
  {
    kL1Norm,
    kL2Norm,
    kLogDifference,
    kFoote,
    kItakuraSaito,
    kKullbackLiebler,
    kSymmetricKullbackLiebler,
    kModifiedKullbackLiebler
  };
  
  OnsetSegmentation()
  {
  }

private:

  double frameComparison(Real& vec1, Real& vec2)
  {
    using namespace descriptors;
    
    if (mForwardOnly)
      Descriptors::forwardFilter(vec1, vec2);
    
    if (mNormalisation != kNone)
    {
      Descriptors::normalise(vec1, mNormalisation == kPower);
      Descriptors::normalise(vec2, mNormalisation == kPower);
    }
    
    switch (mFunction)
    {
      case kL1Norm:                      return Descriptors::differenceL1Norm(vec1, vec2);
      case kL2Norm:                      return Descriptors::differenceL2Norm(vec1, vec2);
      case kLogDifference:               return Descriptors::differenceLog(vec1, vec2);
      case kFoote:                       return Descriptors::differenceFT(vec1, vec2);
      case kItakuraSaito:                return Descriptors::differenceIS(vec1, vec2);
      case kKullbackLiebler:             return Descriptors::differenceKL(vec1, vec2);
      case kSymmetricKullbackLiebler:    return Descriptors::differenceSKL(vec1, vec2);
      case kModifiedKullbackLiebler:     return Descriptors::differenceMKL(vec1, vec2);
    }
  }
    
private:

  Normalisation mNormalisation;
  bool mForwardOnly;
  DifferenceFunction mFunction;
};

};  // namespace segmentation
};  // namespace fluid
