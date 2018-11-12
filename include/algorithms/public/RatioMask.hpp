#pragma once

#include "../../data/FluidTensor.hpp"
#include "../../data/TensorTypes.hpp"
#include "../util/FluidEigenMappings.hpp"
#include <Eigen/Dense>

namespace fluid {
namespace algorithm {

using Eigen::ArrayXXcd;
using Eigen::ArrayXXd;
using Eigen::Map;

class RatioMask {
  const double epsilon = std::numeric_limits<double>::epsilon();

public:
  RatioMask(RealMatrix denominator, int exponent) : mExponent(exponent) {
    ArrayXXdMap denominatorArray(denominator.data(), denominator.extent(0),
                                 denominator.extent(1));
    mMultiplier = (1 / denominatorArray.max(epsilon));
  }

  void process(const ComplexMatrix &mixture, RealMatrix targetMag,
               ComplexMatrix result) {
    assert(mixture.cols() == targetMag.cols());
    assert(mixture.rows() == targetMag.rows());
    // ComplexMatrix result(mixture.extent(0), mixture.extent(1));
    ArrayXXcdConstMap mixtureArray(mixture.data(), mixture.extent(0),
                                   mixture.extent(1));
    ArrayXXdMap targetMagArray(targetMag.data(), targetMag.extent(0),
                               targetMag.extent(1));
    ArrayXXcd tmp =
        mixtureArray *
        (targetMagArray.pow(mExponent) * mMultiplier.pow(mExponent)).min(1.0);
    ArrayXXcdMap(result.data(), mixture.extent(0), mixture.extent(1)) = tmp;
  }

private:
  ArrayXXd mMultiplier;
  int mExponent;
};

} // namespace algorithm
} // namespace fluid
