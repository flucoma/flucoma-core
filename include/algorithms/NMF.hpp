#pragma once

#include "data/FluidEigenMappings.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <vector>

namespace fluid {
namespace nmf {

using Eigen::Array;
using Eigen::ArrayXd;
using Eigen::ArrayXXd;
using Eigen::Map;
using Eigen::Dynamic;
using Eigen::RowMajor;
using Eigen::MatrixXd;
using Eigen::VectorXd;

using fluid::eigenmappings::FluidToMatrixXd;
using fluid::eigenmappings::MatrixXdToFluid;

struct NMFModel {
  using RealMatrix = FluidTensor<double, 2>;
  using RealVector = FluidTensor<double, 1>;

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

const auto &epsilon = std::numeric_limits<double>::epsilon;

class NMF {
public:
  using RealMatrix = FluidTensor<double, 2>;
  using RealVector = FluidTensor<double, 1>;
  using ArrayXdMap = Map<Array<double, Dynamic, RowMajor>>;
  using ArrayXdConstMap = Map<const Array<double, Dynamic, RowMajor>>;

  NMF(int rank, int nIterations, bool updateW = true, bool updateH = true)
      : mRank(rank), mIterations(nIterations), mUpdateW(updateW),
        mUpdateH(updateH) {}

  // processFrame computes activations of a dictionary W in a given frame
  void processFrame(const RealVector x, const RealMatrix W0, RealVector &out,
                    int nIterations = 10) {
    int rank = W0.extent(1);
    MatrixXd W = FluidToMatrixXd(W0)();
    VectorXd h =
        MatrixXd::Random(rank, 1) * 0.5 + MatrixXd::Constant(rank, 1, 0.5);
    VectorXd v = ArrayXdConstMap(x.data(), x.extent(0)).matrix();

    MatrixXd WT = W.transpose();
    WT.colwise().normalize();

    VectorXd ones = VectorXd::Ones(x.extent(0));
    while (nIterations--) {
      ArrayXd v1 = (W * h).array() + epsilon();
      ArrayXXd hNum = (WT * (v.array() / v1).matrix()).array();
      ArrayXXd hDen = (WT * ones).array();
      h = (h.array() * hNum / hDen.max(epsilon())).matrix();
      // VectorXd r = W * h;
      // double divergence = (v.cwiseProduct(v.cwiseQuotient(r)) - v + r).sum();
      // std::cout<<"Divergence "<<divergence<<std::endl;
    }
    ArrayXdMap(out.data(), rank) = h.array();
  }

  const NMFModel process(const RealMatrix &X, RealMatrix W0 = RealMatrix(0, 0),
                         RealMatrix H0 = RealMatrix(0, 0)) {
    int nFrames = X.extent(0);
    int nBins = X.extent(1);

    MatrixXd W;
    if (W0.extent(0) == 0 && W0.extent(1) == 0) {
      W = MatrixXd::Random(nBins, mRank) * 0.5 +
          MatrixXd::Constant(nBins, mRank, 0.5);
    } else {
      assert(W0.extent(0) == mRank);
      assert(W0.extent(1) == nBins);
      W = FluidToMatrixXd(W0)().transpose();
    }
    MatrixXd H;
    if (H0.extent(0) == 0 && H0.extent(1) == 0) {
      H = MatrixXd::Random(mRank, nFrames) * 0.5 +
          MatrixXd::Constant(mRank, nFrames, 0.5);
    } else {
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

  NMFModel multiplicativeUpdates(const MatrixXd V, MatrixXd W, MatrixXd H) {
    NMFModel result;
    std::vector<double> divergenceCurve;
    MatrixXd ones = MatrixXd::Ones(V.rows(), V.cols());
    W.colwise().normalize();
    H.rowwise().normalize();
    while (mIterations--) {
      if (mUpdateW) {
        ArrayXXd V1 = (W * H).array() + epsilon();
        ArrayXXd Wnum = ((V.array() / V1).matrix() * H.transpose()).array();
        ArrayXXd Wden = (ones * H.transpose()).array();
        W = (W.array() * Wnum / Wden.max(epsilon())).matrix();
        if (W.maxCoeff() > epsilon())
          W.colwise().normalize();
        assert(W.allFinite());
      }
      ArrayXXd V2 = (W * H).array() + epsilon();
      if (mUpdateH) {
        ArrayXXd Hnum = (W.transpose() * (V.array() / V2).matrix()).array();
        ArrayXXd Hden = (W.transpose() * ones).array();
        H = (H.array() * Hnum / Hden.max(epsilon())).matrix();
        assert(H.allFinite());
      }
      MatrixXd R = W * H;
      R = R.cwiseMax(epsilon());
      // double divergence = (V.cwiseProduct(V.cwiseQuotient(R)) - V + R).sum();
      // divergenceCurve.push_back(divergence);
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
