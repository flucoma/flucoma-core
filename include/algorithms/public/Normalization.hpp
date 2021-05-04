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

#include "../util/FluidEigenMappings.hpp"
#include "../util/AlgorithmUtils.hpp"
#include "../../data/TensorTypes.hpp"
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
    mMin = min;
    mMax = max;
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
      result = (input - mDataMin) / mDataRange.max(epsilon);
      result = mMin + (result * (mMax - mMin));
    }
    else {
      result = (input - mMin) / std::max((mMax - mMin), epsilon);
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
      result = result.rowwise() / mDataRange.transpose().max(epsilon);
      result = mMin + (result * (mMax - mMin));
    }
    else {
      result = input - mMin;
      result = result / std::max((mMax - mMin), epsilon);
      result = (result.rowwise() * mDataRange.transpose());
      result = (result.rowwise() + mDataMin.transpose());
    }
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

  void clear() {
    mMin = 0;
    mMax = 1.0;
    mDataMin.setZero();
    mDataMax.setZero();
    mDataRange.setZero();
    mInitialized = false;
  }

  double mMin{0.0};
  double mMax{1.0};
  ArrayXd mDataMin;
  ArrayXd mDataMax;
  ArrayXd mDataRange;
  bool mInitialized{false};
};
}; // namespace algorithm
}; // namespace fluid
