#pragma once

#include <Eigen/Dense>
#include "data/FluidTensor.hpp"

namespace fluid {
namespace ratiomask {

using Eigen::ArrayXXcd;
using Eigen::ArrayXXd;
using Eigen::Map;
using RealMatrix = FluidTensor<double, 2>;
using ComplexMatrix = FluidTensor<std::complex<double>, 2>;

using ArrayXXdMap =
    Map<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

using ArrayXXcdMap = Map<Eigen::Array<std::complex<double>, Eigen::Dynamic,
                                      Eigen::Dynamic, Eigen::RowMajor>>;

const auto &epsilon = std::numeric_limits<double>::epsilon;

class RatioMask {
public:
  RatioMask(RealMatrix denominator, int exponent) : mExponent(exponent) {
    ArrayXXdMap denominatorArray(denominator.data(), denominator.extent(0),
                                 denominator.extent(1));
    mMultiplier = (1 / denominatorArray).max(epsilon());
  }

  ComplexMatrix process(ComplexMatrix mixture, RealMatrix targetMag) {
    assert(mixture.cols() == targetMag.cols());
    assert(mixture.rows() == targetMag.rows());
    ComplexMatrix result(mixture.extent(0), mixture.extent(1));
    ArrayXXcdMap mixtureArray(mixture.data(), mixture.extent(0),
                              mixture.extent(1));
    ArrayXXdMap targetMagArray(targetMag.data(), targetMag.extent(0),
                               targetMag.extent(1));
    ArrayXXcd tmp = mixtureArray * (targetMagArray.pow(mExponent) *
                                    mMultiplier.pow(mExponent));
    ArrayXXcdMap(result.data(), mixture.extent(0), mixture.extent(1)) = tmp;
    return result;
  }

private:
  ArrayXXd mMultiplier;
  int mExponent;
};

} // namespace ratiomask
} // namespace fluid
