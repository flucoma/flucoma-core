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
#include <cassert>
#include <queue>
#include <random>
#include <string>

namespace fluid {
namespace algorithm {

class SKMeans : public KMeans
{
  using MatrixLike = Eigen::Ref<const Eigen::MatrixXd>; 
public:

  using KMeans::InitMethod; 

  void train(const FluidDataSet<std::string, double, 1>& dataset, index k,
             index maxIter, InitMethod initialize, index seed)
  {
    using namespace Eigen;
    using namespace _impl;
    assert(!mTrained || (dataset.pointSize() == mDims && mK == k));
    MatrixXd dataPoints =
        asEigen<Matrix>(dataset.getData()).rowwise().normalized();
    if (mTrained) { mAssignments = assignClusters(dataPoints.transpose());}
    else
    {
      mK = k;
      mDims = dataset.pointSize();
      initMeans(dataPoints, initialize, seed);
    }

    while (maxIter-- > 0)
    {
      mEmbedding.noalias() = mMeans.matrix() * dataPoints.transpose();      
      auto assignments = assignClusters(mEmbedding);
      if (mAssignments.rows() && !changed(assignments)) { break; }
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
  void initMeans(Eigen::MatrixXd& dataPoints, InitMethod init, index seed)
  {
    using namespace Eigen;
    mMeans = ArrayXXd::Zero(mK, mDims);

    using namespace _impl::kmeans_init;
    switch(init)
    {
      case InitMethod::randomSampling: 
      {
        mMeans = akmc2(dataPoints, mK, squareCosine, seed);
        break; 
      }
      case InitMethod::randomPoint: 
      {
          mMeans = randomPoints(dataPoints, mK, seed); 
          break; 
      }
      default: { 
        mMeans = randomPartition(dataPoints, mK, seed); 
        mMeans.matrix().rowwise().normalize(); 
      }
    }    
  }

  void updateEmbedding()
  {
    for (index i = 0; i < mAssignments.rows(); i++)
    {
      mEmbedding.col(i).setZero();
      mEmbedding(mAssignments(i), i) = 1.0;
    }
  }


  Eigen::VectorXi
  assignClusters(MatrixLike const& embedding) const
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


  void computeMeans(MatrixLike const& dataPoints)
  {
    mMeans.matrix().noalias() = mEmbedding * dataPoints;
    mMeans.matrix().rowwise().normalize();
  }


private:
  Eigen::MatrixXd mEmbedding;
};
} // namespace algorithm
} // namespace fluid
