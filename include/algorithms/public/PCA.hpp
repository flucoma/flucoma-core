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
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>
#include <Eigen/SVD>
#include <cassert>
#include <cmath>

namespace fluid {
namespace algorithm {

class PCA
{
public:
  using MatrixXd = Eigen::MatrixXd;
  using VectorXd = Eigen::VectorXd;

  void init(RealMatrixView in)
  {
    using namespace Eigen;
    using namespace _impl;
    MatrixXd input = asEigen<Matrix>(in);
    mMean = input.colwise().mean();
    MatrixXd         X = (input.rowwise() - mMean.transpose());
    BDCSVD<MatrixXd> svd(X.matrix(), ComputeThinV | ComputeThinU);
    mBases = svd.matrixV();
    mValues = svd.singularValues();
    mInitialized = true;
  }

  void init(RealMatrixView bases, RealVectorView values, RealVectorView mean)
  {
    mBases = _impl::asEigen<Eigen::Matrix>(bases);
    mValues = _impl::asEigen<Eigen::Matrix>(values);
    mMean = _impl::asEigen<Eigen::Matrix>(mean);
    mInitialized = true;
  }

  void processFrame(const RealVectorView in, RealVectorView out, index k) const
  {
    using namespace Eigen;
    using namespace _impl;
    if (k > mBases.cols()) return;
    VectorXd input = asEigen<Matrix>(in);
    input = input - mMean;
    VectorXd result = input.transpose() * mBases.block(0, 0, mBases.rows(), k);
    out <<= _impl::asFluid(result);
  }

  void inverseProcessFrame(RealVectorView in, RealVectorView out) const
  {
    using namespace Eigen;
    using namespace _impl;
    if(in.size() != dims() || out.size() != dims()) return;
    asEigen<Matrix>(out) = mMean + (asEigen<Matrix>(in).transpose() * mBases.transpose()).transpose();
  }
    
      
  double process(const RealMatrixView in, RealMatrixView out, index k) const
  {
    using namespace Eigen;
    using namespace _impl;
    if (k > mBases.cols()) return 0;
    MatrixXd input = asEigen<Matrix>(in);
    MatrixXd result = (input.rowwise() - mMean.transpose()) *
                      mBases.block(0, 0, mBases.rows(), k);
    double variance = 0;
    double total = mValues.sum();
    for (index i = 0; i < k; i++) variance += mValues[i];
    out <<= _impl::asFluid(result);
    return variance / total;
  }

  void inverseProcess(RealMatrixView in, RealMatrixView out) const
  {
    using namespace Eigen;

    if (in.cols() > dims()) return;
    if (out.cols() < in.cols()) return;
    _impl::asEigen<Matrix>(out) =
        (_impl::asEigen<Matrix>(in) * mBases.transpose()).rowwise() +
        mMean.transpose();
  }

  bool  initialized() const { return mInitialized; }
  void  getBases(RealMatrixView out) const { out <<= _impl::asFluid(mBases); }
  void  getValues(RealVectorView out) const { out <<= _impl::asFluid(mValues); }
  void  getMean(RealVectorView out) const { out <<= _impl::asFluid(mMean); }
  index dims() const { return mBases.rows(); }
  index size() const { return mBases.cols(); }
  void  clear()
  {
    mBases.setZero();
    mMean.setZero();
    mInitialized = false;
  }

  MatrixXd mBases;
  VectorXd mValues;
  VectorXd mMean;
  bool     mInitialized{false};
};
}// namespace algorithm
}// namespace fluid
