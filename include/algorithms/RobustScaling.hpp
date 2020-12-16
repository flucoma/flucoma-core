//modified version of Normalization.hpp code
#pragma once

#include "algorithms/util/FluidEigenMappings.hpp"
#include "data/TensorTypes.hpp"

#include <Eigen/Core>
#include <cassert>
#include <cmath>

namespace fluid {
namespace algorithm {

class RobustScaling {
public:
  using ArrayXd = Eigen::ArrayXd;
  using ArrayXXd = Eigen::ArrayXXd;

void init(double min, double max, double low, double high, RealMatrixView in) {
    using namespace Eigen;
    using namespace _impl;
    mMin = min;
    mMax = max;
    mLow = low;
    mHigh = high;
    ArrayXXd input = asEigen<Array>(in);
    mDataMin = input.row(0);//stupid init (maybe name change to mDataLow)
    mDataMax = input.row(0);//stupid init (maybe name change to mDataHigh)
    //iterating through the colums, I'm sure there is a cleaner/faster way
    index length = input.rows();//stupid init
    for (index i = 0;i<input.cols();i++) {
      ArrayXd sorted = input.col(i);
      std::sort(sorted.data(), sorted.data() + length);
      mDataMin(i) = sorted(lrint((mLow / 100.0) * (length - 1)));
      mDataMax(i) = sorted(lrint((mHigh / 100.0) * (length - 1)));
    }
    mDataRange = mDataMax - mDataMin;
    mInitialized = true;
  }

void init(double min, double max, double low, double high, RealVectorView dataMin,
            RealVectorView dataMax) {
    using namespace Eigen;
    using namespace _impl;
    const double epsilon = std::numeric_limits<double>::epsilon();
    mMin = min;
    mMax = max;
    mLow = low;
    mHigh = high;
    mDataMin = asEigen<Array>(dataMin);
    mDataMax = asEigen<Array>(dataMax);
    mDataRange = mDataMax - mDataMin;
    mDataRange = mDataRange.max(epsilon);
    mInitialized = true;
  }

  void processFrame(const RealVectorView in, RealVectorView out, bool inverse = false) const{
    using namespace Eigen;
    using namespace _impl;
    ArrayXd input = asEigen<Array>(in);
    ArrayXd result;
    if(!inverse) {
      result = (input - mDataMin) / mDataRange;
      result = mMin + (result * (mMax - mMin));
    }
    else {
      result = (input - mMin) / (mMax - mMin);
      result = mDataMin + (result * mDataRange);
    }
    out = asFluid(result);
  }

  void process(const RealMatrixView in, RealMatrixView out, bool inverse = false) const{
    using namespace Eigen;
    using namespace _impl;
    ArrayXXd input = asEigen<Array>(in);
    ArrayXXd result;
    if(!inverse) {
      result = (input.rowwise() - mDataMin.transpose());
      result = result.rowwise() / mDataRange.transpose();
      result = mMin + (result * (mMax - mMin));
    }
    else {
      result = input - mMin;
      result = result / (mMax - mMin);
      result = (result.rowwise() * mDataRange.transpose());
      result = (result.rowwise() + mDataMin.transpose());
    }
    out = asFluid(result);
  }

  void setMin(double min) { mMin = min; }
  void setMax(double max) { mMax = max; }
  void setLow(double low) { mLow = low; }
  void setHigh(double high) { mHigh = high; }
  bool initialized() const{ return mInitialized; }

  double getMin() const{ return mMin; }
  double getMax() const{ return mMax; }
  double getLow() const{ return mLow; }
  double getHigh() const{ return mHigh; }

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

  void clear() {
    mMin = 0;
    mMax = 1.0;
    mLow = 0;
    mHigh = 1.0;
    mDataMin.setZero();
    mDataMax.setZero();
    mDataRange.setZero();
    mInitialized = false;
  }

  double mMin{0.0};
  double mMax{1.0};
  double mLow{0.0};
  double mHigh{1.0};
  ArrayXd mDataMin;
  ArrayXd mDataMax;
  ArrayXd mDataRange;
  bool mInitialized{false};
};
}; // namespace algorithm
}; // namespace fluid
