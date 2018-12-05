#pragma once

#include "../../data/TensorTypes.hpp"
#include "../util/FluidEigenMappings.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <vector>

namespace fluid {
namespace algorithm {

using _impl::asEigen;
using _impl::asFluid;
using Eigen::ArrayXd;
using Eigen::ArrayXXd;
using Eigen::MatrixXd;
using Eigen::VectorXd;

class NMF {
  double const epsilon = std::numeric_limits<double>::epsilon();

public:
  NMF(int rank, int nIterations, bool updateW = true, bool updateH = true)
      : mRank(rank), mIterations(nIterations), mUpdateW(updateW),
        mUpdateH(updateH) {}

  static void estimate(const RealMatrix W, const RealMatrix H, int index,
                       RealMatrix V) {
    MatrixXd W1 = asEigen<Matrix>(W).transpose();
    MatrixXd H1 = asEigen<Matrix>(H).transpose();
    MatrixXd result = (W1.col(index) * H1.row(index)).transpose();
    V = asFluid(result);
  }
  // TODO: use newer way to map FluidView for rows/columns
  /*
  static void estimate(const RealVector W, const RealVector H, RealMatrix V){
    MatrixXd w = ArrayXdConstMap(W.data(), W.extent(0)).matrix();
    MatrixXd h = ArrayXdConstMap(H.data(), H.extent(0)).matrix();
    ArrayXXdMap outV = ArrayXXdMap(V.data(), V.extent(0), V.extent(1));
    ArrayXXd v1 = (w * h.transpose()).transpose().array();
    outV = v1;
  }
  */

  // processFrame computes activations of a dictionary W in a given frame
  void processFrame(const RealVector x, const RealMatrix W0, RealVector out,
                    int nIterations = 10) {
    MatrixXd W = asEigen<Matrix>(W0);
    VectorXd h =
        MatrixXd::Random(mRank, 1) * 0.5 + MatrixXd::Constant(mRank, 1, 0.5);
    VectorXd v = asEigen<Matrix>(x);
    MatrixXd WT = W.transpose();
    WT.colwise().normalize();
    VectorXd ones = VectorXd::Ones(x.extent(0));
    while (nIterations--) {
      ArrayXd v1 = (W * h).array() + epsilon;
      ArrayXXd hNum = (WT * (v.array() / v1).matrix()).array();
      ArrayXXd hDen = (WT * ones).array();
      h = (h.array() * hNum / hDen.max(epsilon)).matrix();
      // VectorXd r = W * h;
      // double divergence = (v.cwiseProduct(v.cwiseQuotient(r)) - v + r).sum();
      // std::cout<<"Divergence "<<divergence<<std::endl;
    }
    out = asFluid(h);
    // ArrayXdMap(out.data(), mRank) = h.array();
  }

  void process(const RealMatrix X, RealMatrix W1, RealMatrix H1, RealMatrix V1,
               RealMatrix W0 = RealMatrix(0, 0),
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
      W = asEigen<Matrix>(W0).transpose();
    }
    MatrixXd H;
    if (H0.extent(0) == 0 && H0.extent(1) == 0) {
      H = MatrixXd::Random(mRank, nFrames) * 0.5 +
          MatrixXd::Constant(mRank, nFrames, 0.5);
    } else {
      assert(H0.extent(0) == nFrames);
      assert(H0.extent(1) == mRank);
      H = asEigen<Matrix>(H0).transpose();
    }
    MatrixXd V = asEigen<Matrix>(X).transpose();
    multiplicativeUpdates(V, W, H);
    MatrixXd VT = V.transpose();
    MatrixXd WT = W.transpose();
    MatrixXd HT = H.transpose();
    V1 = asFluid(VT);
    W1 = asFluid(WT);
    H1 = asFluid(HT);
  }

private:
  int mRank;
  int mIterations;
  bool mUpdateW;
  bool mUpdateH;

  void multiplicativeUpdates(MatrixXd &V, MatrixXd &W, MatrixXd &H) {
    MatrixXd ones = MatrixXd::Ones(V.rows(), V.cols());
    W.colwise().normalize();
    H.rowwise().normalize();
    while (mIterations--) {
      if (mUpdateW) {
        ArrayXXd V1 = (W * H).array() + epsilon;
        ArrayXXd wnum = ((V.array() / V1).matrix() * H.transpose()).array();
        ArrayXXd wden = (ones * H.transpose()).array();
        W = (W.array() * wnum / wden.max(epsilon)).matrix();
        if (W.maxCoeff() > epsilon)
          W.colwise().normalize();
        assert(W.allFinite());
      }
      ArrayXXd V2 = (W * H).array() + epsilon;
      if (mUpdateH) {
        ArrayXXd hnum = (W.transpose() * (V.array() / V2).matrix()).array();
        ArrayXXd hden = (W.transpose() * ones).array();
        H = (H.array() * hnum / hden.max(epsilon)).matrix();
        assert(H.allFinite());
      }
      MatrixXd R = W * H;
      R = R.cwiseMax(epsilon);
      double divergence = (V.cwiseProduct(V.cwiseQuotient(R)) - V + R).sum();
      // divergenceCurve.push_back(divergence);
      // divergenceCurve(mIterations);
      // std::cout << "Divergence " << divergence << "\n";
    }
    V = W * H;
  }
};
} // namespace algorithm
} // namespace fluid
