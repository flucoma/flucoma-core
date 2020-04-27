#pragma once

#include "algorithms/util/FluidEigenMappings.hpp"
#include "data/TensorTypes.hpp"

#include <Eigen/Core>
#include <Eigen/SVD>
#include <cassert>
#include <cmath>

namespace fluid {
namespace algorithm {

class PCA {
public:
  using MatrixXd = Eigen::MatrixXd;
  using VectorXd = Eigen::VectorXd;

  void init(RealMatrixView in, index k) {
    using namespace Eigen;
    using namespace _impl;
    MatrixXd input = asEigen<Matrix>(in);
    mMean = input.colwise().mean();
    MatrixXd X = (input.rowwise() - mMean.transpose());
    BDCSVD<MatrixXd> svd(X.matrix(), ComputeThinV | ComputeThinU);
    MatrixXd V = svd.matrixV();
    mBases = V.block(0, 0, V.rows(), k);
    mInitialized = true;
  }

  void init(RealMatrixView bases, RealVectorView mean) {
    mBases = _impl::asEigen<Eigen::Matrix>(bases);
    mMean = _impl::asEigen<Eigen::Matrix>(mean);
    mInitialized = true;
  }

  void processFrame(const RealVectorView in, RealVectorView out) const {
    using namespace Eigen;
    using namespace _impl;
    VectorXd input = asEigen<Matrix>(in);
    input = input - mMean;
    VectorXd result = input.transpose() * mBases;
    out = _impl::asFluid(result);
  }

  void process(const RealMatrixView in, RealMatrixView out) const {
    using namespace Eigen;
    using namespace _impl;
    MatrixXd input = asEigen<Matrix>(in);
    MatrixXd result = (input.rowwise() - mMean.transpose()) * mBases;
    out = _impl::asFluid(result);
  }

  bool initialized() const { return mInitialized; }
  void getBases(RealMatrixView out) const { out = _impl::asFluid(mBases); }
  void getMean(RealVectorView out) const { out = _impl::asFluid(mMean); }

  MatrixXd mBases;
  VectorXd mMean;
  bool mInitialized{false};
};
}; // namespace algorithm
}; // namespace fluid
