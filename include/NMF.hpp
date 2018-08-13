#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <vector>

namespace fluid {
namespace nmf {

using Eigen::ArrayXXd;
using Eigen::Map;
using Eigen::MatrixXd;
// using fluid::FluidTensor;
using std::cout;
using std::vector;

using RealMatrix = FluidTensor<double, 2>;
using RealVector = FluidTensor<double, 1>;
using MatrixXdMap =
    Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

struct NMFModel {
  MatrixXd W;
  MatrixXd H;
  MatrixXd V;
  RealVector divergence;
  RealMatrix getEstimate(int index) const {
    assert(index < W.cols());
    RealMatrix result(H.cols(), W.rows()); // transposing
    MatrixXdMap(result.data(), H.cols(), W.rows()) =
        (W.col(index) * H.row(index)).transpose();
    return result;
  }

  RealMatrix getMixEstimate() const {
    RealMatrix result(H.cols(), W.rows());// transposing
    MatrixXdMap(result.data(), H.cols(), W.rows()) = (W * H).transpose();
    return result;
  }

  RealMatrix getW() const {
    RealMatrix result(W.rows(), W.cols());
    MatrixXdMap(result.data(), W.rows(), W.cols()) = W;
    return result;
  }

  RealMatrix getH() const {
    RealMatrix result(H.rows(), H.cols());
    MatrixXdMap(result.data(), H.rows(), H.cols()) = H;
    return result;
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
    MatrixXdMap Vorig(X.data(), X.extent(0), X.extent(1));
    MatrixXd V = Vorig.transpose();
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
      // cout << "Divergence " << divergence << "\n";
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
