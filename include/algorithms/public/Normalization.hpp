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

#include "../util/ScalerUtils.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>
#include <cassert>
#include <cmath>

namespace fluid {
namespace algorithm {

class Normalization
{
public:
  using ArrayXd = Eigen::ArrayXd;
  using ArrayXXd = Eigen::ArrayXXd;

  void init(double min, double max, RealMatrixView in)
  {
    using namespace Eigen;
    using namespace _impl;
    mMin = min;
    mMax = max;
    mRange = mMax - mMin;
    handleZerosInScale(mRange);
    ArrayXXd input = asEigen<Array>(in);
    mDataMin = input.colwise().minCoeff();
    mDataMax = input.colwise().maxCoeff();
    mDataRange = mDataMax - mDataMin;
    handleZerosInScale(mDataRange);
    mInitialized = true;
  }

  void init(double min, double max, RealVectorView dataMin,
            RealVectorView dataMax)
  {
    using namespace Eigen;
    using namespace _impl;
    mMin = min;
    mMax = max;
    mRange = mMax - mMin;
    handleZerosInScale(mRange);
    mDataMin = asEigen<Array>(dataMin);
    mDataMax = asEigen<Array>(dataMax);
    mDataRange = mDataMax - mDataMin;
    handleZerosInScale(mDataRange);
    mInitialized = true;
  }

  void processFrame(const RealVectorView in, RealVectorView out,
                    bool inverse = false) const
  {
    using namespace Eigen;
    using namespace _impl;
    FluidEigenMap<Array> input  = asEigen<Array>(in);
    FluidEigenMap<Array> result = asEigen<Array>(out);
    if (!inverse)
    {
      result = (input - mDataMin) / mDataRange;
      result = mMin + (result * mRange);
    }
    else
    {
      result = (input - mMin) / mRange;
      result = mDataMin + (result * mDataRange);
    }
  }

  void process(const RealMatrixView in, RealMatrixView out,
               bool inverse = false) const
  {
    using namespace Eigen;
    using namespace _impl;
    ArrayXXd input = asEigen<Array>(in);
    ArrayXXd result;
    if (!inverse)
    {
      result = (input.rowwise() - mDataMin.transpose());
      result = result.rowwise() / mDataRange.transpose();
      result = mMin + (result * mRange);
    }
    else
    {
      result = input - mMin;
      result = result / mRange;
      result = (result.rowwise() * mDataRange.transpose());
      result = (result.rowwise() + mDataMin.transpose());
    }
    out <<= asFluid(result);
  }

  void setMin(double min) { 
    mMin = min; 
    mRange = mMax - mMin;
    handleZerosInScale(mRange);
  }
  void setMax(double max) { 
    mMax = max;
    mRange = mMax - mMin;
    handleZerosInScale(mRange);
  }
  bool initialized() const { return mInitialized; }

  double getMin() const { return mMin; }
  double getMax() const { return mMax; }

  void getDataMin(RealVectorView out) const
  {
    using namespace _impl;
    out <<= asFluid(mDataMin);
  }

  void getDataMax(RealVectorView out) const
  {
    using namespace _impl;
    out <<= asFluid(mDataMax);
  }

  index dims() const { return mDataMin.size(); }
  index size() const { return 1; }

  void clear()
  {
    mMin = 0;
    mMax = 1.0;
    mRange = 1.0;
    mDataMin.setZero();
    mDataMax.setZero();
    mDataRange.setZero();
    mInitialized = false;
  }

  double  mMin{0.0};
  double  mMax{1.0};
  double  mRange{1.0};
  ArrayXd mDataMin;
  ArrayXd mDataMax;
  ArrayXd mDataRange;
  bool    mInitialized{false};
};
}// namespace algorithm
}// namespace fluid
