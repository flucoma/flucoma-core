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

#include "../util/DistanceFuncs.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../../data/FluidDataSet.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/FluidTensor.hpp"
#include "../../data/TensorTypes.hpp"
#include "../../data/FluidMemory.hpp"
#include <Eigen/Core>
#include <queue>
#include <string>

namespace fluid {
namespace algorithm {

class KMeans
{

public:
  void clear()
  {
    mMeans.setZero();
    mAssignments.setZero();
    mTrained = false;
  }

  bool initialized() const { return mTrained; }

  void train(const FluidDataSet<std::string, double, 1>& dataset, index k,
             index maxIter)
  {
    using namespace Eigen;
    using namespace _impl;
    assert(!mTrained || (dataset.pointSize() == mDims && mK == k));
    auto dataPoints = asEigen<Array>(dataset.getData());
    if (mTrained) { mAssignments = assignClusters(dataPoints); }
    else
    {
      mK = k;
      mDims = dataset.pointSize();
      mMeans = ArrayXXd::Zero(mK, mDims);
      mEmpty = std::vector<bool>(asUnsigned(mK), false);
      mAssignments =
          ((0.5 + (0.5 * ArrayXf::Random(dataPoints.rows()))) * (mK - 1))
              .round()
              .cast<int>();
    }

    while (maxIter-- > 0)
    {
      computeMeans(dataPoints);
      auto assignments = assignClusters(dataPoints);
      if (!changed(assignments)) { break; }
      else
      {
        mAssignments = assignments;
      }
    }
    mTrained = true;
  }

  index getClusterSize(index cluster) const
  {
    index count = 0;
    for (index i = 0; i < mAssignments.size(); i++)
    {
      if (mAssignments(i) == cluster) count++;
    }
    return count;
  }

  index vq(RealVectorView point) const
  {
    assert(point.size() == mDims);
    // transpose() allows us to avoid a temporary further down the call stack
    return assignPoint(_impl::asEigen<Eigen::Array>(point).transpose());
  }

  void getMeans(RealMatrixView out) const
  {
    if (mTrained) out <<= _impl::asFluid(mMeans);
  }

  void setMeans(RealMatrixView means)
  {
    mMeans = _impl::asEigen<Eigen::Array>(means);
    mDims = mMeans.cols();
    mK = mMeans.rows();
    mEmpty = std::vector<bool>(asUnsigned(mK), false);
    mTrained = true;
  }

  index dims() const { return mMeans.cols(); }
  index size() const { return mMeans.rows(); }
  index getK() const { return mMeans.rows(); }
  index nAssigned() const { return mAssignments.size(); }

  void getAssignments(FluidTensorView<index, 1> out) const
  {
    out <<= _impl::asFluid(mAssignments);
  }

  void transform(RealMatrixView data, RealMatrixView out) const
  {
    Eigen::ArrayXXd points = _impl::asEigen<Eigen::Array>(data);
    Eigen::ArrayXXd D = fluid::algorithm::DistanceMatrix(points, 2);
    Eigen::MatrixXd means = mMeans.matrix();
    D = fluid::algorithm::DistanceMatrix<Eigen::ArrayXXd>(points, mMeans, 2);
    out <<= _impl::asFluid(D);
  }

protected:
  template <typename DerivedA, typename DerivedB>
  double distance(const Eigen::ArrayBase<DerivedA>& v1,
                  const Eigen::ArrayBase<DerivedB>& v2) const
  {
    return (v1 - v2).matrix().norm();
  }

  template <typename Derived>
  index assignPoint(const Eigen::ArrayBase<Derived>& point) const
  {
    double minDistance = std::numeric_limits<double>::infinity();
    index  minK;
    for (index k = 0; k < mK; k++)
    {
      double dist = distance(point, mMeans.row(k));
      if (dist < minDistance)
      {
        minK = k;
        minDistance = dist;
      }
    }
    return minK;
  }

  Eigen::VectorXi assignClusters(const Eigen::ArrayXXd& dataPoints) const
  {
    Eigen::VectorXi assignments = Eigen::VectorXi::Zero(dataPoints.rows());
    for (index i = 0; i < dataPoints.rows(); i++)
    {
      assignments(i) = static_cast<int>(assignPoint(dataPoints.row(i)));
    }
    return assignments;
  }

  void computeMeans(const Eigen::ArrayXXd& dataPoints)
  {
    using namespace Eigen;
    for (index k = 0; k < mK; k++)
    {
      if (mEmpty[asUnsigned(k)]) continue;
      std::vector<index> kAssignment;
      for (index i = 0; i < mAssignments.size(); i++)
      {
        if (mAssignments(i) == k) kAssignment.push_back(i);
      }
      if (kAssignment.size() == 0)
      {
        std::cout << "Warning: empty cluster" << std::endl;
        mEmpty[asUnsigned(k)] = true;
        return;
      }
      ArrayXXd clusterPoints =
          ArrayXXd::Zero(asSigned(kAssignment.size()), mDims);
      for (index i = 0; asUnsigned(i) < kAssignment.size(); i++)
      {
        clusterPoints.row(i) = dataPoints.row(kAssignment[asUnsigned(i)]);
      }
      ArrayXd mean = clusterPoints.colwise().mean();
      mMeans.row(k) = mean;
    }
  }

  bool changed(const Eigen::VectorXi& newAssignments) const
  {
    auto dif = (newAssignments - mAssignments).cwiseAbs().sum();
    return dif > 0;
  }

  index             mK{0};
  index             mDims{0};
  Eigen::ArrayXXd   mMeans;
  std::vector<bool> mEmpty;
  Eigen::VectorXi   mAssignments;
  bool              mTrained{false};
};
} // namespace algorithm
} // namespace fluid
