#pragma once

#include "algorithms/util/FluidEigenMappings.hpp"
#include "data/TensorTypes.hpp"

#include <Eigen/Core>
#include <cassert>
#include <cmath>

namespace fluid {
namespace algorithm {

class Standardization {
public:
  using ArrayXd = Eigen::ArrayXd;
  using ArrayXXd = Eigen::ArrayXXd;

  void init(RealMatrixView in) {
    using namespace Eigen;
    using namespace _impl;
    ArrayXXd input = asEigen<Array>(in);
    mMean = input.colwise().mean();
    mStd = ((input.rowwise() - mMean.transpose()).square().colwise().mean());
    mInitialized = true;
  }

  void init(const RealVectorView mean, const RealVectorView std) {
    using namespace Eigen;
    using namespace _impl;
    mMean = asEigen<Array>(mean);
    mStd = asEigen<Array>(std);
    mInitialized = true;
  }

  void processFrame(const RealVectorView in, RealVectorView out) const{
    using namespace Eigen;
    using namespace _impl;
    ArrayXd input = asEigen<Array>(in);
    ArrayXd result = (input - mMean) / mStd;
    out = asFluid(result);
  }

  void process(const RealMatrixView in, RealMatrixView out) const{
    using namespace Eigen;
    using namespace _impl;
    ArrayXXd input = asEigen<Array>(in);
    ArrayXXd result = (input.rowwise() - mMean.transpose());
    result = result.rowwise() / mStd.transpose();
    out = asFluid(result);
  }

  bool initialized() const{ return mInitialized; }

  void getMean(RealVectorView out) const{ out = _impl::asFluid(mMean); }

  void getStd(RealVectorView out) const{ out = _impl::asFluid(mStd); }

  ArrayXd mMean;
  ArrayXd mStd;
  bool mInitialized{false};
};
}; // namespace algorithm
}; // namespace fluid
