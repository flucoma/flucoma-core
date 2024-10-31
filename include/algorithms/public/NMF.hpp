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

#include "../util/AlgorithmUtils.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/TensorTypes.hpp"
#include "../../data/FluidMemory.hpp"
#include <Eigen/Core>
#include <vector>

namespace fluid {
namespace algorithm {

class NMF
{

public:
  // pass iteration number; returns true if able to continue (i.e. not
  // cancelled)
  using ProgressCallback = std::function<bool(index)>;

  static void estimate(const RealMatrixView W, const RealMatrixView H,
                       index idx, RealMatrixView V)
  {
    using namespace Eigen;
    using namespace _impl;

    auto W1 = asEigen<Matrix>(W).transpose();
    auto H1 = asEigen<Matrix>(H).transpose();
    asEigen<Matrix>(V).transpose().noalias() = (W1.col(idx) * H1.row(idx));
  }

  // processFrame computes activations of a dictionary W in a given frame
  void processFrame(const RealVectorView x, const RealMatrixView W0,
                    RealVectorView out, index nIterations,
                    RealVectorView v, Allocator& alloc)
  {
    using namespace Eigen;
    using namespace _impl;
    index    rank = W0.extent(0);
    FluidEigenMap<Matrix> W = asEigen<Matrix>(W0);
    
    ScopedEigenMap<VectorXd> h(rank, alloc);
    h = VectorXd::Random(rank) * 0.5 + VectorXd::Constant(rank, 0.5);
    
    ScopedEigenMap<VectorXd> v0(x.size(), alloc);
    v0 = asEigen<Matrix>(x);
    W = W.array().max(epsilon).matrix();
    h = h.array().max(epsilon).matrix();
    v0 = v0.array().max(epsilon).matrix();

//    MatrixXd WT = W.transpose();
    ScopedEigenMap<VectorXd> norm(W.rowwise().norm(), alloc);
    W.array().colwise() /= norm.array();
    index nBins = x.extent(0);

    ScopedEigenMap<ArrayXd> v1{nBins, alloc};
    ScopedEigenMap<ArrayXd> vRatio{nBins, alloc};
    ScopedEigenMap<ArrayXd> hNum{rank, alloc};
    ScopedEigenMap<ArrayXd> hDen{rank, alloc};
    auto ones = VectorXd::Ones(nBins);
    while (nIterations--)
    {
      v1.matrix().noalias() = (W.transpose() * h);
      v1 = v1.max(epsilon);
      vRatio = v0.array() / v1;
      hNum.matrix().noalias() = W * vRatio.matrix();
      hDen.matrix().noalias() = W * ones;
      h = (h.array() * hNum / hDen.max(epsilon)).matrix();
      // VectorXd r = W * h;
      // double divergence = (v.cwiseProduct(v.cwiseQuotient(r)) - v + r).sum();
      // std::cout<<"Divergence "<<divergence<<std::endl;
    }
    
    if(out.data())
      _impl::asEigen<Array>(out) =  h;

    if (v.data()) _impl::asEigen<Matrix>(v).noalias() = (W.transpose() * h);
  }

  void process(const RealMatrixView X, RealMatrixView W1, RealMatrixView H1,
               RealMatrixView V1, index rank, index nIterations, bool updateW,
               bool           updateH = false,
               RealMatrixView W0 = RealMatrixView(nullptr, 0, 0, 0),
               RealMatrixView H0 = RealMatrixView(nullptr, 0, 0, 0))
  {
    using namespace Eigen;
    using namespace _impl;
    index    nFrames = X.extent(0);
    index    nBins = X.extent(1);
    MatrixXd W;
    if (W0.extent(0) == 0 && W0.extent(1) == 0)
    {
      W = MatrixXd::Random(nBins, rank) * 0.5 +
          MatrixXd::Constant(nBins, rank, 0.5);
    }
    else
    {
      assert(W0.extent(0) == rank);
      assert(W0.extent(1) == nBins);
      W = asEigen<Matrix>(W0).transpose();
    }
    MatrixXd H;
    if (H0.extent(0) == 0 && H0.extent(1) == 0)
    {
      H = MatrixXd::Random(rank, nFrames) * 0.5 +
          MatrixXd::Constant(rank, nFrames, 0.5);
    }
    else
    {
      assert(H0.extent(0) == nFrames);
      assert(H0.extent(1) == rank);
      H = asEigen<Matrix>(H0).transpose();
    }
    MatrixXd V = asEigen<Matrix>(X).transpose();
    multiplicativeUpdates(V, W, H, nIterations, updateW, updateH);
    MatrixXd VT = V.transpose();
    MatrixXd WT = W.transpose();
    MatrixXd HT = H.transpose();

    V1 <<= asFluid(VT);
    W1 <<= asFluid(WT);
    H1 <<= asFluid(HT);
  }

  void addProgressCallback(ProgressCallback&& callback)
  {
    mCallbacks.emplace_back(std::move(callback));
  }

private:
  using MatrixXd = Eigen::MatrixXd;

  void multiplicativeUpdates(Eigen::Ref<MatrixXd> V, Eigen::Ref<MatrixXd> W,
                             Eigen::Ref<MatrixXd> H, index nIterations,
                             bool updateW, bool updateH)
  {
    using namespace Eigen;
    MatrixXd ones = MatrixXd::Ones(V.rows(), V.cols());
    H = H.array().max(epsilon).matrix();
    W = W.array().max(epsilon).matrix();
    W.colwise().normalize();
    H.rowwise().normalize();
    for (auto i = 0; i < nIterations; ++i)
    {
      if (updateW)
      {
        ArrayXXd V1 = (W * H).array().max(epsilon);
        ArrayXXd wnum = ((V.array() / V1).matrix() * H.transpose()).array();
        ArrayXXd wden = (ones * H.transpose()).array();
        W = (W.array() * wnum / wden.max(epsilon)).matrix();
        if (W.maxCoeff() > epsilon) W.colwise().normalize();
        assert(W.allFinite());
      }
      ArrayXXd V2 = (W * H).array().max(epsilon);
      if (updateH)
      {
        ArrayXXd hnum = (W.transpose() * (V.array() / V2).matrix()).array();
        ArrayXXd hden = (W.transpose() * ones).array();
        H = (H.array() * hnum / hden.max(epsilon)).matrix();
        assert(H.allFinite());
      }
      MatrixXd R = W * H;
      R = R.cwiseMax(epsilon);
      for (auto& cb : mCallbacks)
        if (!cb(i + 1)) return;
      // double divergence = (V.cwiseProduct(V.cwiseQuotient(R)) - V + R).sum();
      // divergenceCurve.push_back(divergence);
      // divergenceCurve(mIterations);
      // std::cout << "Divergence " << divergence << "\n";
    }
    V = W * H;
  }

  std::vector<ProgressCallback> mCallbacks;
};
} // namespace algorithm
} // namespace fluid
