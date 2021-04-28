#pragma once

#include "algorithms/util/DistanceFuncs.hpp"
#include "algorithms/util/FluidEigenMappings.hpp"
#include "data/TensorTypes.hpp"

#include <Eigen/Core>
#include <Eigen/SVD>
#include <cassert>
#include <cmath>

namespace fluid {
namespace algorithm {

class MDS {
public:
  using MatrixXd = Eigen::MatrixXd;
  using VectorXd = Eigen::VectorXd;

  void process(RealMatrixView in, RealMatrixView out, index distance, index k) {
    using namespace Eigen;
    using namespace _impl;
    auto dist = static_cast<DistanceFuncs::Distance>(distance);
    MatrixXd input = asEigen<Matrix>(in);
    index n = input.rows();
    MatrixXd D = MatrixXd::Zero(n, n);
    MatrixXd I = MatrixXd::Identity(n, n);
    MatrixXd ones = MatrixXd::Ones(n, n);
    for (index i = 0; i < n; i++) {
      for (index j = 0; j < n; j++) {
        D(i, j) = DistanceFuncs::map()[dist](input.row(i), input.row(j));
      }
    }
    MatrixXd J = I - ones / n;
    D = -0.5 * J * D * J;
    BDCSVD<MatrixXd> svd(D, ComputeThinV | ComputeThinU);
    MatrixXd U = svd.matrixU();
    ArrayXd s = svd.singularValues().segment(0, k);
    MatrixXd result = U.block(0, 0, U.rows(), k).array().rowwise() *s.transpose();
    out = asFluid(result);
  }
};
}; // namespace algorithm
}; // namespace fluid
