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
#include <Eigen/Core>
#include <Eigen/SVD>

namespace fluid {
namespace algorithm {

class NNDSVD
{

public:
  using MatrixXd = Eigen::MatrixXd;

  index process(RealMatrixView X, RealMatrixView W, RealMatrixView H,
                index minRank = 0, index maxRank = 200, double amount = 0.8,
                index method = 0) // 0 - NMF-SVD, 1 NNDSVDar, 2 NNDSVDa 3 NNDSVD
  {
    using namespace _impl;
    using namespace Eigen;
    MatrixXd XT = asEigen<Matrix>(X).transpose();
    MatrixXd WT = asEigen<Matrix>(W).transpose();
    MatrixXd HT = asEigen<Matrix>(H).transpose();

    assert(amount > 0 || minRank > 0);

    BDCSVD<MatrixXd> svd(XT, ComputeThinV | ComputeThinU);

    MatrixXd U = svd.matrixU();
    MatrixXd V = svd.matrixV().transpose();
    VectorXd s = svd.singularValues();
    MatrixXd S = svd.singularValues().asDiagonal();
    index    k = 0;
    if (amount == 0)
      k = minRank;
    else
    {
      double current = 0;
      double total = s.sum();
      while ((current / total) < amount) current += s[k++];
    }
    if (k < minRank) k = minRank;
    if (k > maxRank) k = maxRank;

    if (method == 0)
    {
      WT.block(0, 0, WT.rows(), k) = U.block(0, 0, U.rows(), k).array().abs();
      HT.block(0, 0, k, HT.cols()) =
          (S.block(0, 0, k, S.cols()) * V).array().abs();
    }
    else
    {
      // avoid scaling for NMF with normalized W
      WT.col(0) = U.col(0).array().abs();
      HT.row(0) = sqrt(s(0)) * V.row(0).array().abs();

      for (index j = 1; j < k; j++)
      {
        VectorXd x = U.col(j);
        VectorXd y = V.row(j);
        VectorXd xP = x.array().max(0.0);
        VectorXd yP = y.array().max(0.0);
        VectorXd xN = x.array().min(0.0).abs();
        VectorXd yN = y.array().min(0.0).abs();
        double   xPNorm = xP.norm();
        double   yPNorm = yP.norm();
        double   xNNorm = xN.norm();
        double   yNNorm = xN.norm();
        double   mP = xPNorm * yPNorm;
        double   mN = xNNorm * yNNorm;
        ArrayXd  u;
        ArrayXd  v;
        double   sigma;
        if (mP > mN)
        {
          u = xP / xPNorm;
          v = yP / yPNorm;
          sigma = mP;
        }
        else
        {
          u = xN / xNNorm;
          v = yN / yNNorm;
          sigma = mN;
        }
        auto lbd = std::sqrt(s[j] * sigma);
        WT.col(j) = u; // avoid scaling for NMF with normalized W
        HT.row(j) = lbd * v;
      }
      WT = WT.array().max(epsilon);
      HT = HT.array().max(epsilon);
      if (method == 1)
      {
        auto Wrand =
            MatrixXd::Random(WT.rows(), WT.cols()).array().abs() / 100.0;
        auto Hrand =
            MatrixXd::Random(HT.rows(), HT.cols()).array().abs() / 100.0;
        WT = (WT.array() < epsilon).select(Wrand, WT);
        HT = (HT.array() < epsilon).select(Hrand, HT);
      }
      else if (method == 2)
      {
        double mean = XT.mean();
        WT = (WT.array() < epsilon)
                 .select(MatrixXd::Constant(WT.rows(), WT.cols(), mean), WT);
        HT = (HT.array() < epsilon)
                 .select(MatrixXd::Constant(HT.rows(), HT.cols(), mean), HT);
      }
    }
    MatrixXd W1 = WT.transpose();
    W <<= asFluid(W1);
    MatrixXd H1 = HT.transpose();
    H <<= asFluid(H1);
    return k;
  }
};
} // namespace algorithm
} // namespace fluid
