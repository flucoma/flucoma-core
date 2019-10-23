/*
Copyright 2017-2019 University of Huddersfield.
Licensed under the BSD-3 License.
See LICENSE file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/

#pragma once

#include "../util/AlgorithmUtils.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>
#include <vector>

namespace fluid {
namespace algorithm {

class NMF
{

public:
  // pass iteration number; returns true if able to continue (i.e. not
  // cancelled)
  using ProgressCallback = std::function<bool(int)>;

  NMF(int maxRank) : mRank(maxRank) {}

  void init(int rank, int nIterations, bool updateW = true, bool updateH = true)
  {
    mRank = rank;
    mIterations = nIterations;
    mUpdateW = updateW;
    mUpdateH = updateH;
  }

  static void estimate(const RealMatrixView W, const RealMatrixView H,
                       int index, RealMatrixView V)
  {
    using namespace Eigen;
    using namespace _impl;

    MatrixXd W1 = asEigen<Matrix>(W).transpose();
    MatrixXd H1 = asEigen<Matrix>(H).transpose();
    MatrixXd result = (W1.col(index) * H1.row(index)).transpose();
    V = asFluid(result);
  }

  // processFrame computes activations of a dictionary W in a given frame
  void processFrame(const RealVectorView x, const RealMatrixView W0,
                    RealVectorView out, int nIterations = 10,
                    RealVectorView v = RealVectorView(nullptr, 0, 0))
  {
    using namespace Eigen;
    using namespace _impl;
    int      rank = W0.extent(0);
    MatrixXd W = asEigen<Matrix>(W0).transpose();
    VectorXd h =
        MatrixXd::Random(rank, 1) * 0.5 + MatrixXd::Constant(rank, 1, 0.5);
    VectorXd v0 = asEigen<Matrix>(x);
    W = W.array().max(epsilon).matrix();
    h = h.array().max(epsilon).matrix();
    v0 = v0.array().max(epsilon).matrix();

    MatrixXd WT = W.transpose();
    W.colwise().normalize();
    VectorXd ones = VectorXd::Ones(x.extent(0));
    while (nIterations--)
    {
      ArrayXd  v1 = (W * h).array().max(epsilon);
      ArrayXXd hNum = (WT * (v0.array() / v1).matrix()).array();
      ArrayXXd hDen = (WT * ones).array();
      h = (h.array() * hNum / hDen.max(epsilon)).matrix();
      // VectorXd r = W * h;
      // double divergence = (v.cwiseProduct(v.cwiseQuotient(r)) - v + r).sum();
      // std::cout<<"Divergence "<<divergence<<std::endl;
    }
    out = asFluid(h);
    if (v.extent(0) > 0)
    {
      ArrayXd v2 = (W * h).array();
      v = asFluid(v2);
    }
    // ArrayXdMap(out.data(), rank) = h.array();
  }

  void process(const RealMatrixView X, RealMatrixView W1, RealMatrixView H1,
               RealMatrixView V1,
               RealMatrixView W0 = RealMatrixView(nullptr, 0, 0, 0),
               RealMatrixView H0 = RealMatrixView(nullptr, 0, 0, 0))
  {
    using namespace Eigen;
    using namespace _impl;
    int      nFrames = X.extent(0);
    int      nBins = X.extent(1);
    MatrixXd W;
    if (W0.extent(0) == 0 && W0.extent(1) == 0)
    {
      W = MatrixXd::Random(nBins, mRank) * 0.5 +
          MatrixXd::Constant(nBins, mRank, 0.5);
    }
    else
    {
      assert(W0.extent(0) == mRank);
      assert(W0.extent(1) == nBins);
      W = asEigen<Matrix>(W0).transpose();
    }
    MatrixXd H;
    if (H0.extent(0) == 0 && H0.extent(1) == 0)
    {
      H = MatrixXd::Random(mRank, nFrames) * 0.5 +
          MatrixXd::Constant(mRank, nFrames, 0.5);
    }
    else
    {
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

  void addProgressCallback(ProgressCallback&& callback)
  {
    mCallbacks.emplace_back(std::move(callback));
  }

private:
  using MatrixXd = Eigen::MatrixXd;

  void multiplicativeUpdates(MatrixXd& V, MatrixXd& W, MatrixXd& H)
  {
    using namespace Eigen;
    MatrixXd ones = MatrixXd::Ones(V.rows(), V.cols());
    H = H.array().max(epsilon).matrix();
    W = W.array().max(epsilon).matrix();
    W.colwise().normalize();
    H.rowwise().normalize();
    for (auto i = 0; i < mIterations; ++i)
    {
      if (mUpdateW)
      {
        ArrayXXd V1 = (W * H).array().max(epsilon);
        ArrayXXd wnum = ((V.array() / V1).matrix() * H.transpose()).array();
        ArrayXXd wden = (ones * H.transpose()).array();
        W = (W.array() * wnum / wden.max(epsilon)).matrix();
        if (W.maxCoeff() > epsilon) W.colwise().normalize();
        assert(W.allFinite());
      }
      ArrayXXd V2 = (W * H).array().max(epsilon);
      if (mUpdateH)
      {
        ArrayXXd hnum = (W.transpose() * (V.array() / V2).matrix()).array();
        ArrayXXd hden = (W.transpose() * ones).array();
        H = (H.array() * hnum / hden.max(epsilon)).matrix();
        assert(H.allFinite());
      }
      MatrixXd R = W * H;
      R = R.cwiseMax(epsilon);
      double divergence = (V.cwiseProduct(V.cwiseQuotient(R)) - V + R).sum();
      for (auto& cb : mCallbacks)
        if (!cb(i + 1)) return;
      // divergenceCurve.push_back(divergence);
      // divergenceCurve(mIterations);
      // std::cout << "Divergence " << divergence << "\n";
    }
    V = W * H;
  }

  int  mRank{1};
  int  mIterations{100};
  bool mUpdateW{true};
  bool mUpdateH{true};

  std::vector<ProgressCallback> mCallbacks;
};
} // namespace algorithm
} // namespace fluid
