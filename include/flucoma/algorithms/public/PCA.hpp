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
  using ArrayXd = Eigen::ArrayXd;

  void init(RealMatrixView in)
  {
    using namespace Eigen;
    using namespace _impl;
    MatrixXd input = asEigen<Matrix>(in);
    mNumDataPoints = input.rows();
    mMean = input.colwise().mean();
    MatrixXd         X = (input.rowwise() - mMean.transpose());
    BDCSVD<MatrixXd> svd(X.matrix(), ComputeThinV | ComputeThinU);
    mBases = svd.matrixV();
    mValues = svd.singularValues();
    mExplainedVariance = mValues.array().square() / (mNumDataPoints - 1);
    mInitialized = true;
  }

  void init(RealMatrixView bases, RealVectorView values, RealVectorView mean,
            index numDataPoints = 2)
  {
    mBases = _impl::asEigen<Eigen::Matrix>(bases);
    mValues = _impl::asEigen<Eigen::Matrix>(values);
    mMean = _impl::asEigen<Eigen::Matrix>(mean);
    mNumDataPoints = numDataPoints;
    mExplainedVariance = mValues.array().square() / (mNumDataPoints - 1);
    mInitialized = true;
  }

  void processFrame(const RealVectorView in, RealVectorView out, index k,
                    bool whiten = false) const
  {
    using namespace Eigen;
    using namespace _impl;
    if (k > mBases.cols()) return;
    FluidEigenMap<Matrix> input = asEigen<Matrix>(in);
    input = input - mMean;
    FluidEigenMap<Matrix> result = asEigen<Matrix>(out);
    // this avoids Eigen making a temporary (sorry):
    result.transpose().noalias() =
        input.transpose() * mBases.block(0, 0, mBases.rows(), k);

    if (whiten)
    {
      auto norm = mExplainedVariance.segment(0, k).max(epsilon).rsqrt();
      result.array() *= norm;
    }
  }

  void inverseProcessFrame(RealVectorView in, RealVectorView out, bool whiten = false) const
  {
    using namespace Eigen;
    using namespace _impl;
    
    if(!whiten)
    {
      asEigen<Matrix>(out) =
          mMean +
          (asEigen<Matrix>(in).transpose() * mBases.transpose()).transpose();
    }
    else
    {
      asEigen<Matrix>(out) = mMean +  (asEigen<Matrix>(in).transpose() *
            (mExplainedVariance.sqrt().matrix().asDiagonal() * mBases.transpose())).transpose();
    }    
  }

  double process(const RealMatrixView in, RealMatrixView out, index k,
                 bool whiten = false) const
  {
    using namespace Eigen;
    using namespace _impl;

    if (k > mBases.cols()) return 0;
    MatrixXd input = asEigen<Matrix>(in);
    MatrixXd result = (input.rowwise() - mMean.transpose()) *
                      mBases.block(0, 0, mBases.rows(), k);
    if (whiten)
    {
      ArrayXd norm = mExplainedVariance.segment(0, k).max(epsilon).rsqrt();
      result = result.array().rowwise() * norm.transpose().max(epsilon);
    }
    double variance = 0;

    double total = mExplainedVariance.sum();
    for (index i = 0; i < k; i++) variance += mExplainedVariance[i];
    out <<= _impl::asFluid(result);

    return variance / total;
  }

  void inverseProcess(RealMatrixView in, RealMatrixView out, bool whiten = false) const
  {
    using namespace Eigen;

    if (in.cols() > dims()) return;
    if (out.cols() < in.cols()) return;

    if (!whiten)
      _impl::asEigen<Matrix>(out) =
          (_impl::asEigen<Matrix>(in) * mBases.transpose()).rowwise() +
          mMean.transpose();

    else
    {
      _impl::asEigen<Matrix>(out) =
          (_impl::asEigen<Matrix>(in) *
           (mExplainedVariance.sqrt().matrix().asDiagonal() *
            mBases.transpose()))
              .rowwise() +
          mMean.transpose();
    }
  }

  bool  initialized() const { return mInitialized; }

  void  getBases(RealMatrixView out) const { out <<= _impl::asFluid(mBases); }
  void  getValues(RealVectorView out) const { out <<= _impl::asFluid(mValues); }
  void  getMean(RealVectorView out) const { out <<= _impl::asFluid(mMean); }
  index getNumDataPoints() const { return mNumDataPoints; }

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
  ArrayXd  mExplainedVariance;
  VectorXd mMean;
  index    mNumDataPoints;
  bool     mInitialized{false};
};
}// namespace algorithm
}// namespace fluid
