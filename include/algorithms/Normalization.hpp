#pragma once

#include "algorithms/util/FluidEigenMappings.hpp"
#include "data/TensorTypes.hpp"

#include <Eigen/Core>
#include <cassert>
#include <cmath>

namespace fluid {
namespace algorithm {

class Normalization {
public:
  using ArrayXd = Eigen::ArrayXd;
  using ArrayXXd = Eigen::ArrayXXd;

  void init(double min, double max, RealMatrixView in) {
    using namespace Eigen;
    using namespace _impl;
    mMin = min;
    mMax = max;
    ArrayXXd input = asEigen<Array>(in);
    mDataMin = input.colwise().minCoeff();
    mDataMax = input.colwise().maxCoeff();
    mDataRange = mDataMax - mDataMin;
    mInitialized = true;
  }

  void init(double min, double max, RealVectorView dataMin,
            RealVectorView dataMax) {
    using namespace Eigen;
    using namespace _impl;
    const double epsilon = std::numeric_limits<double>::epsilon();
    mMin = min;
    mMax = max;
    mDataMin = asEigen<Array>(dataMin);
    mDataMax = asEigen<Array>(dataMax);
    mDataRange = mDataMax - mDataMin;
    mDataRange = mDataRange.max(epsilon);
    mInitialized = true;
  }

  void processFrame(const RealVectorView in, RealVectorView out) const{
    using namespace Eigen;
    using namespace _impl;
    ArrayXd input = asEigen<Array>(in);
    ArrayXd result = (input - mDataMin) / mDataRange;
    result = mMin + (result * (mMax - mMin));
    out = asFluid(result);
  }

  void process(const RealMatrixView in, RealMatrixView out) const{
    using namespace Eigen;
    using namespace _impl;
    ArrayXXd input = asEigen<Array>(in);
    ArrayXXd result = (input.rowwise() - mDataMin.transpose());
    result = result.rowwise() / mDataRange.transpose();
    result = mMin + (result * (mMax - mMin));
    out = asFluid(result);
  }

  void setMin(double min) { mMin = min; }
  void setMax(double max) { mMax = max; }
  bool initialized() const{ return mInitialized; }

  double getMin() const{ return mMin; }
  double getMax() const{ return mMax; }

  void getDataMin(RealVectorView out) const{
    using namespace _impl;
    out = asFluid(mDataMin);
  }

  void getDataMax(RealVectorView out) const{
    using namespace _impl;
    out = asFluid(mDataMax);
  }

  index dims() const { return mDataMin.size(); }
  index size() const { return 1;}

  double mMin{0.0};
  double mMax{1.0};
  ArrayXd mDataMin;
  ArrayXd mDataMax;
  ArrayXd mDataRange;
  bool mInitialized{false};
};
}; // namespace algorithm
}; // namespace fluid
