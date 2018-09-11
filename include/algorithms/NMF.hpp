#pragma once

#include "data/FluidEigenMappings.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <vector>

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
    return MatrixXdToFluid(W.transpose())();
  }

  RealMatrix getH() const {
    RealMatrix result(H.rows(), H.cols());
    return MatrixXdToFluid(H.transpose())();
  }
};

class NMF {
public:
  NMF(int rank, int nIterations, bool updateW = true, bool updateH = true)
      : mRank(rank), mIterations(nIterations), mUpdateW(updateW),
        mUpdateH(updateH) {}

  const NMFModel process(const RealMatrix &X, RealMatrix W0 = RealMatrix(0, 0),
                         RealMatrix H0 = RealMatrix(0, 0)) {
    int nFrames = X.extent(0);
    int nBins = X.extent(1);

    MatrixXd W;
    if (W0.extent(0) == 0 && W0.extent(1) == 0) {
      W = MatrixXd::Random(nBins, mRank) * 0.5 +
          MatrixXd::Constant(nBins, mRank, 0.5);
    } else{
      assert(W0.extent(0) == mRank);
      assert(W0.extent(1) == nBins);
      W = FluidToMatrixXd(W0)().transpose();
    }
    MatrixXd H;
    if (H0.extent(0) == 0 && H0.extent(1) == 0) {
      H = MatrixXd::Random(mRank, nFrames) * 0.5 +
                   MatrixXd::Constant(mRank, nFrames, 0.5);
    } else{
      assert(H0.extent(0) == nFrames);
      assert(H0.extent(1) == mRank);
        H = FluidToMatrixXd(H0)().transpose();
    }

    MatrixXd V = FluidToMatrixXd(X)().transpose();
    return multiplicativeUpdates(V, W, H);
  }

private:
  int mRank;
  int mIterations;
  bool mUpdateW;
  bool mUpdateH;

  NMFModel multiplicativeUpdates(const MatrixXd V, MatrixXd W, MatrixXd H){
    const auto &epsilon = std::numeric_limits<double>::epsilon;
    NMFModel result;
    vector<double> divergenceCurve;
    MatrixXd ones = MatrixXd::Ones(V.rows(), V.cols());
    W.colwise().normalize();
    H.rowwise().normalize();
    while (mIterations--) {
      if (mUpdateW) {
        ArrayXXd V1 = (W * H).array() + epsilon();
        ArrayXXd Wnum = ((V.array() / V1).matrix() * H.transpose()).array();
        ArrayXXd Wden = (ones * H.transpose()).array();
        W = (W.array() * Wnum / Wden.max(epsilon())).matrix();
        W.colwise().normalize();
      }
      ArrayXXd V2 = (W * H).array() + epsilon();
      if (mUpdateH) {
        ArrayXXd Hnum = (W.transpose() * (V.array() / V2).matrix()).array();
        ArrayXXd Hden = (W.transpose() * ones).array();
        H = (H.array() * Hnum / Hden.max(epsilon())).matrix();
      }
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



};
} // namespace nmf
} // namespace fluid
