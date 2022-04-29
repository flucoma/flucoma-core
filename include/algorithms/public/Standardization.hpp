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

#include "../util/AlgorithmUtils.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>
#include <cassert>
#include <cmath>

namespace fluid {
namespace algorithm {

class Standardization
{
public:
  using ArrayXd = Eigen::ArrayXd;
  using ArrayXXd = Eigen::ArrayXXd;

  void init(RealMatrixView in)
  {
    using namespace Eigen;
    using namespace _impl;
    ArrayXXd input = asEigen<Array>(in);
    mMean = input.colwise().mean();
    mStd = ((input.rowwise() - mMean.transpose()).square().colwise().mean())
               .sqrt();
    mInitialized = true;
  }

  void init(const RealVectorView mean, const RealVectorView std)
  {
    using namespace Eigen;
    using namespace _impl;
    mMean = asEigen<Array>(mean);
    mStd = asEigen<Array>(std);
    mInitialized = true;
  }

  void processFrame(const RealVectorView in, RealVectorView out,
                    bool inverse = false) const
  {
    using namespace Eigen;
    using namespace _impl;
    ArrayXd input = asEigen<Array>(in);
    ArrayXd result;
    if (!inverse) { result = (input - mMean) / mStd.max(epsilon); }
    else
    {
      result = (input * mStd) + mMean;
    }
    out <<= asFluid(result);
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
      result = (input.rowwise() - mMean.transpose());
      result = result.rowwise() / mStd.transpose().max(epsilon);
    }
    else
    {
      result = (input.rowwise() * mStd.transpose());
      result = (result.rowwise() + mMean.transpose());
    }
    out <<= asFluid(result);
  }

  bool initialized() const { return mInitialized; }

  void getMean(RealVectorView out) const { out <<= _impl::asFluid(mMean); }

  void getStd(RealVectorView out) const { out <<= _impl::asFluid(mStd); }

  index dims() const { return mMean.size(); }
  index size() const { return 1; }

  void clear()
  {
    mMean.setZero();
    mStd.setZero();
    mInitialized = false;
  }

  ArrayXd mMean;
  ArrayXd mStd;
  bool    mInitialized{false};
};
}// namespace algorithm
}// namespace fluid
