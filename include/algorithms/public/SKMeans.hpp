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

#include "../public/KMeans.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../../data/FluidDataSet.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/FluidTensor.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>
#include <queue>
#include <string>

namespace fluid {
namespace algorithm {

class SKMeans : public KMeans
{

public:
  void train(const FluidDataSet<std::string, double, 1>& dataset, index k,
             index maxIter)
  {
    using namespace Eigen;
    using namespace _impl;
    assert(!mTrained || (dataset.pointSize() == mDims && mK == k));
    MatrixXd dataPoints = asEigen<Matrix>(dataset.getData());
    MatrixXd dataPointsT = dataPoints.transpose();
    if (mTrained) { mAssignments = assignClusters(dataPointsT);}
    else
    {
      mK = k;
      mDims = dataset.pointSize();
      initMeans(dataPoints);
    }

    while (maxIter-- > 0)
    {
      mEmbedding = mMeans.matrix() * dataPointsT;
      auto assignments = assignClusters(mEmbedding);
      if (!changed(assignments)) { break; }
      else
        mAssignments = assignments;
      updateEmbedding();
      computeMeans(dataPoints);
    }
    mTrained = true;
  }


  void encode(RealMatrixView data, RealMatrixView out,
                 double alpha = 0.25) const
  {
    using namespace Eigen;
    MatrixXd points = _impl::asEigen<Matrix>(data).transpose();
    MatrixXd embedding = (mMeans.matrix() * points).array() - alpha;
    embedding = (embedding.array() > 0).select(embedding, 0).transpose();
    out <<= _impl::asFluid(embedding);
  }

private:

  void initMeans(Eigen::MatrixXd& dataPoints)
  {
    using namespace Eigen;
    mMeans = ArrayXXd::Zero(mK, mDims);
    mAssignments =
        ((0.5 + (0.5 * ArrayXd::Random(dataPoints.rows()))) * (mK - 1))
            .round()
            .cast<int>();
    mEmbedding = MatrixXd::Zero(mK, dataPoints.rows());
    for (index i = 0; i < dataPoints.rows(); i++)
      mEmbedding(mAssignments(i), i) = 1;
    computeMeans(dataPoints);
  }

  void updateEmbedding()
  {
    for (index i = 0; i < mAssignments.cols(); i++)
    {
      double val = mEmbedding(mAssignments(i), i);
      mEmbedding.col(i).setZero();
      mEmbedding(mAssignments(i), i) = val;
    }
  }


  Eigen::VectorXi assignClusters(Eigen::MatrixXd& embedding) const
  {
    Eigen::VectorXi assignments = Eigen::VectorXi::Zero(embedding.cols());
    for (index i = 0; i < embedding.cols(); i++)
    {
      Eigen::VectorXd::Index maxIndex;
      embedding.col(i).maxCoeff(&maxIndex);
      assignments(i) = static_cast<int>(maxIndex);
    }
    return assignments;
  }


  void computeMeans(Eigen::MatrixXd& dataPoints)
  {
    mMeans = mEmbedding * dataPoints;
    mMeans.matrix().rowwise().normalize();
  }


private:
  Eigen::MatrixXd mEmbedding;
};
} // namespace algorithm
} // namespace fluid
