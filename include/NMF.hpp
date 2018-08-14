#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <FluidEigenMappings.hpp>

namespace fluid {
namespace nmf {

using Eigen::ArrayXXd;
using Eigen::Map;
using Eigen::MatrixXd;
using std::vector;

using fluid::eigenmappings::FluidToMatrixXd;
using fluid::eigenmappings::MatrixXdToFluid;


using RealMatrix = FluidTensor<double, 2>;
using RealVector = FluidTensor<double, 1>;

struct NMFModel {
  MatrixXd W;
  MatrixXd H;
  MatrixXd V;
  RealVector divergence;
  RealMatrix getEstimate(int index) const {
    assert(index < W.cols());
    return MatrixXdToFluid((W.col(index) * H.row(index)).transpose())();
  }

  RealMatrix getMixEstimate() const {
    RealMatrix result(H.cols(), W.rows());
    return MatrixXdToFluid((W * H).transpose())();
  }

  RealMatrix getW() const {
    RealMatrix result(W.rows(), W.cols());
    return MatrixXdToFluid(W)();
  }

  RealMatrix getH() const {
    RealMatrix result(H.rows(), H.cols());
    return MatrixXdToFluid(H)();
  }
};

class NMF {
public:
  NMF(int rank, int nIterations) : mRank(rank), mIterations(nIterations) {}

  const NMFModel process(const RealMatrix &X) {
    const auto &epsilon = std::numeric_limits<double>::epsilon;
    NMFModel result;
    int nFrames = X.extent(0);
    int nBins = X.extent(1);
    MatrixXd ones = MatrixXd::Ones(nBins, nFrames);
    MatrixXd W = MatrixXd::Random(nBins, mRank) * 0.5 +
                 MatrixXd::Constant(nBins, mRank, 0.5);
    MatrixXd H = MatrixXd::Random(mRank, nFrames) * 0.5 +
                 MatrixXd::Constant(mRank, nFrames, 0.5);
    // TODO: could we transpose X in place? Does not seem possible with Map
    MatrixXd V = FluidToMatrixXd(X)().transpose();
    vector<double> divergenceCurve;
    W.colwise().normalize();
    H.rowwise().normalize();
    while (mIterations--) {
      ArrayXXd V1 = (W * H).array() + epsilon();
      ArrayXXd Wnum = ((V.array() / V1).matrix() * H.transpose()).array();
      ArrayXXd Wden = (ones * H.transpose()).array();
      W = (W.array() * Wnum / Wden.max(epsilon())).matrix();
      W.colwise().normalize();
      ArrayXXd V2 = (W * H).array() + epsilon();
      ArrayXXd Hnum = (W.transpose() * (V.array() / V2).matrix()).array();
      ArrayXXd Hden = (W.transpose() * ones).array();
      H = (H.array() * Hnum / Hden.max(epsilon())).matrix();
      MatrixXd R = W * H;
      R = R.cwiseMax(epsilon());
      MatrixXd V3 = V2.cwiseMax(epsilon());
      double divergence = (V3.cwiseProduct(V3.cwiseQuotient(R)) - V3 + R).sum();
      divergenceCurve.push_back(divergence);
      // std::cout << "Divergence " << divergence << "\n";
    }
    result.W = W;
    result.H = H;
    result.divergence = RealVector(divergenceCurve);
    return result;
  }

private:
  int mRank;
  int mIterations;
};
} // namespace nmf
} // namespace fluid
