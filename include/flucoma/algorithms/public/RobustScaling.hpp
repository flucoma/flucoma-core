/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/

// modified version of Normalization.hpp code
#pragma once

#include "../util/ScalerUtils.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>
#include <cassert>
#include <cmath>

namespace fluid {
namespace algorithm {

class RobustScaling
{
public:
  using ArrayXd = Eigen::ArrayXd;
  using ArrayXXd = Eigen::ArrayXXd;

  void init(double low, double high, RealMatrixView in)
  {
    using namespace Eigen;
    using namespace _impl;
    mLow = low;
    mHigh = high;
    ArrayXXd input = asEigen<Array>(in);
    mDataLow.resize(input.cols());
    mDataHigh.resize(input.cols());
    mMedian.resize(input.cols());
    mRange.resize(input.cols());
    index length = input.rows();
    for (index i = 0; i < input.cols(); i++)
    {
      ArrayXd sorted = input.col(i);
      std::sort(sorted.data(), sorted.data() + length);
      mMedian(i) = sorted(lrint(0.5 * (length - 1)));
      mDataLow(i) = sorted(lrint((mLow / 100.0) * (length - 1)));
      mDataHigh(i) = sorted(lrint((mHigh / 100.0) * (length - 1)));
    }
    mRange = mDataHigh - mDataLow;
    handleZerosInScale(mRange);
    mInitialized = true;
  }

  void init(double low, double high, RealVectorView dataLow,
            RealVectorView dataHigh, RealVectorView median,
            RealVectorView range)
  {
    using namespace Eigen;
    using namespace _impl;
    mLow = low;
    mHigh = high;
    mDataLow = asEigen<Array>(dataLow);
    mDataHigh = asEigen<Array>(dataHigh);
    mMedian = asEigen<Array>(median);
    mRange = asEigen<Array>(range);
    handleZerosInScale(mRange); // in case it is imported from the outside world
    mInitialized = true;
  }

  void processFrame(const RealVectorView in, RealVectorView out,
                    bool inverse = false) const
  {
    using namespace Eigen;
    using namespace _impl;
    FluidEigenMap<Array> input = asEigen<Array>(in);
    FluidEigenMap<Array> result = asEigen<Array>(out);
    if (!inverse) { result = (input - mMedian) / mRange; }
    else
    {
      result = (input * mRange) + mMedian;
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
      result = (input.rowwise() - mMedian.transpose());
      result = result.rowwise() / mRange.transpose();
    }
    else
    {
      result = (input.rowwise() * mRange.transpose());
      result = (result.rowwise() + mMedian.transpose());
    }
    out <<= asFluid(result);
  }

  void setLow(double low) { mLow = low; }
  void setHigh(double high) { mHigh = high; }
  bool initialized() const { return mInitialized; }

  double getLow() const { return mLow; }
  double getHigh() const { return mHigh; }

  void getDataLow(RealVectorView out) const
  {
    using namespace _impl;
    out <<= asFluid(mDataLow);
  }

  void getDataHigh(RealVectorView out) const
  {
    using namespace _impl;
    out <<= asFluid(mDataHigh);
  }

  void getMedian(RealVectorView out) const
  {
    using namespace _impl;
    out <<= asFluid(mMedian);
  }

  void getRange(RealVectorView out) const
  {
    using namespace _impl;
    out <<= asFluid(mRange);
  }

  index dims() const { return mMedian.size(); }
  index size() const { return 1; }

  void clear()
  {
    mLow = 0;
    mHigh = 1.0;
    mMedian.setZero();
    mRange.setZero();
    mRange.setZero();
    mInitialized = false;
  }

  double  mLow{0.0};
  double  mHigh{1.0};
  ArrayXd mDataHigh;
  ArrayXd mDataLow;
  ArrayXd mMedian;
  ArrayXd mRange;
  bool    mInitialized{false};
};
}// namespace algorithm
}// namespace fluid
