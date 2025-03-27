/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/

// Jonathan Driedger, Thomas Prätzlich, and Meinard Müller
// Let It Bee — Towards NMF-Inspired Audio Mosaicing
// Proceedings of ISMIR 2015

#pragma once

#include "STFT.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <vector>

namespace fluid {
namespace algorithm {

using _impl::asEigen;
using _impl::asFluid;
using Eigen::Array;
using Eigen::ArrayXd;
using Eigen::ArrayXXd;
using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::VectorXd;

class NMFCross
{

public:
  // pass iteration number; returns true if able to continue (i.e. not
  // cancelled)
  using ProgressCallback = std::function<bool(index)>;

  NMFCross(index nIterations) : mIterations(nIterations) {}

  static void synthesize(const RealMatrixView h, const ComplexMatrixView w,
                         ComplexMatrixView out)
  {
    using namespace Eigen;
    using namespace _impl;
    MatrixXd  H = asEigen<Matrix>(h);
    MatrixXcd W = asEigen<Matrix>(w);
    MatrixXcd V = H * W;
    out <<= asFluid(V);
  }

  void process(const RealMatrixView X, RealMatrixView H1, RealMatrixView W0,
               index r, index p, index c) const
  {
    index nFrames = X.extent(0);
    index nBins = X.extent(1);
    index rank = W0.extent(0);
    nBins = W0.extent(1);
    MatrixXd W = asEigen<Matrix>(W0).transpose();
    MatrixXd H;
    H = MatrixXd::Random(rank, nFrames) * 0.5 +
        MatrixXd::Constant(rank, nFrames, 0.5);
    MatrixXd V = asEigen<Matrix>(X).transpose();
    multiplicativeUpdates(V, W, H, r, p, c);
    MatrixXd HT = H.transpose();
    H1 <<= asFluid(HT);
  }

  void addProgressCallback(ProgressCallback&& callback)
  {
    mCallbacks.emplace_back(std::move(callback));
  }

private:
  index                         mIterations;
  std::vector<ProgressCallback> mCallbacks;

  std::vector<index> topC(Eigen::VectorXd vec, index c) const
  {
    using namespace std;
    vector<double> stdVec(vec.data(), vec.data() + vec.size());
    sort(stdVec.begin(), stdVec.end());
    vector<index> idx(asUnsigned(vec.size()));
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(),
         [&vec](index i1, index i2) { return vec[i1] > vec[i2]; });
    auto result = std::vector<index>(idx.begin(), idx.begin() + c);
    return result;
  }

  Eigen::MatrixXd promoteContinuity(MatrixXd& H, index size) const
  {
    index    halfSize = (size - 1) / 2;
    MatrixXd kernel = MatrixXd::Identity(size, size);
    MatrixXd padded = MatrixXd::Zero(H.rows() + size, H.cols() + size);
    MatrixXd output = MatrixXd::Zero(H.rows(), H.cols());
    padded.block(halfSize, halfSize, H.rows(), H.cols()) = H;
    for (index i = 0; i < H.rows(); i++)
    {
      for (index j = 0; j < H.cols(); j++)
      {
        output(i, j) =
            padded.block(i, j, size, size).cwiseProduct(kernel).sum();
      }
    }
    return output;
  }

  Eigen::MatrixXd enforceTemporalSparseness(MatrixXd& H, index size,
                                            index iteration) const
  {
    index    halfSize = (size - 1) / 2;
    MatrixXd padded = MatrixXd::Zero(H.rows(), H.cols() + size);
    MatrixXd output = MatrixXd::Zero(H.rows(), H.cols());
    padded.block(0, halfSize, H.rows(), H.cols()) = H;
    for (index i = 0; i < H.rows(); i++)
    {
      for (index j = 0; j < H.cols(); j++)
      {
        VectorXd        neighborhood = padded.row(i).segment(j, size);
        VectorXd::Index maxIndex{0};
        neighborhood.maxCoeff(&maxIndex);
        if (int(maxIndex) != halfSize)
        { output(i, j) = H(i, j) * (1 - ((iteration + 1) / mIterations)); }
        else
        {
          output(i, j) = H(i, j);
        }
      }
    }
    return output;
  }


  Eigen::MatrixXd restrictPolyphony(MatrixXd& H, ArrayXd& energyInW, index size,
                                    index iteration) const
  {
    MatrixXd output = MatrixXd::Zero(H.rows(), H.cols());
    for (index k = 0; k < H.cols(); k++)
    {
      ArrayXd wCol = H.col(k).array() * energyInW.array();
      output.col(k) = H.col(k) * (1 - ((iteration + 1) / mIterations));
      auto top = topC(wCol, size);
      for (auto t : top) { output(t, k) = H(t, k); }
    }
    return output;
  }
  void multiplicativeUpdates(MatrixXd& V, MatrixXd& W, MatrixXd& H, index r,
                             index p, index c) const
  {
    using namespace std;
    using namespace Eigen;
    double const epsilon = std::numeric_limits<double>::epsilon();
    MatrixXd     ones = MatrixXd::Ones(V.rows(), V.cols());
    W = W.array().max(epsilon).matrix();
    // ArrayXd wNorm = W.colwise().sum();
    // W.array().rowwise() /= wNorm.transpose());
    ArrayXd energyInW = W.array().square().colwise().sum();
    for (index i = 0; i < mIterations; i++)
    {
      if ((i % 1) == 0)
      { // TODO: original version seems to work better with one in 5 iterations
        H = enforceTemporalSparseness(H, r, i);
        H = restrictPolyphony(H, energyInW, p, i);
        H = promoteContinuity(H, c);
      }
      ArrayXXd V2 = (W * H).array().max(epsilon);
      ArrayXXd hnum = (W.transpose() * (V.array() / V2).matrix()).array();
      ArrayXXd hden = (W.transpose() * ones).array();
      H = (H.array() * hnum / hden.max(epsilon)).matrix();
      // MatrixXd R = W * H;
      // R = R.cwiseMax(epsilon);
      // double divergence = (V.cwiseProduct(V.cwiseQuotient(R)) - V + R).sum();
      for (auto& cb : mCallbacks)
        if (!cb(i + 1)) return;
    }
    V = W * H;
  }
};
} // namespace algorithm
} // namespace fluid
