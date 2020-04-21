#pragma once

#include "algorithms/util/FluidEigenMappings.hpp"
#include "data/FluidDataSet.hpp"
#include "data/FluidTensor.hpp"
#include "data/TensorTypes.hpp"
#include "data/FluidIndex.hpp"
#include <Eigen/Core>
#include <queue>
#include <string>

namespace fluid {
namespace algorithm {

class KMeans {

public:
  void init(index k, index dims) {
    mK = k;
    mDims = dims;
    mTrained = false;
  }

  bool trained() const { return mTrained; }

  void train(const FluidDataSet<std::string, double, 1> &dataset, index maxIter,
             RealMatrixView initialMeans = RealMatrixView(nullptr, 0, 0, 0)) {
    using namespace Eigen;
    using namespace _impl;
    assert(dataset.pointSize() == mDims);
    auto dataPoints = asEigen<Array>(dataset.getData());
    mEmpty = std::vector<bool>(mK, false);
    if (initialMeans.data() != nullptr) {
      assert(initialMeans.rows() == mK);
      assert(initialMeans.cols() == mDims);
      mMeans = asEigen<Array>(initialMeans);
      mAssignments = assignClusters(dataPoints);
    } else {
      mAssignments =
          ((0.5 + (0.5 * ArrayXf::Random(dataPoints.rows()))) * (mK - 1))
              .round()
              .cast<int>();
      mMeans = ArrayXXd::Zero(mK, mDims);
    }

    while (maxIter-- > 0) {
      computeMeans(dataPoints);
      auto assignments = assignClusters(dataPoints);
      if (!changed(assignments)) {
        break;
      } else {
        mAssignments = assignments;
      }
    }
    mTrained = true;
  }

  size_t getClusterSize(index cluster) const {
    size_t count = 0;
    for (index i = 0; i < mAssignments.size(); i++) {
      if (mAssignments(i) == cluster)
        count++;
    }
    return count;
  }

  index vq(RealVectorView point) const {
    assert(point.size() == mDims);
    return assignPoint(_impl::asEigen<Eigen::Array>(point));
  }

  void getMeans(RealMatrixView out) const {
    if (mTrained)
      out = _impl::asFluid(mMeans);
  }

  void setMeans(RealMatrixView means) {
    mMeans = _impl::asEigen<Eigen::Array>(means);
  }

  index getDims() const { return mDims; }
  index getK() const { return mK; }

  index nAssigned() const { return mAssignments.size(); }

  void getAssignments(FluidTensorView<index, 1> out) const {
    out = _impl::asFluid(mAssignments);
  }

private:
  double distance(Eigen::ArrayXd v1, Eigen::ArrayXd v2) const {
    return (v1 - v2).matrix().norm();
  }

  index assignPoint(Eigen::ArrayXd point) const {
    double minDistance = std::numeric_limits<double>::infinity();
    index minK;
    for (index k = 0; k < mK; k++) {
      double dist = distance(point, mMeans.row(k));
      if (dist < minDistance) {
        minK = k;
        minDistance = dist;
      }
    }
    return minK;
  }

  Eigen::VectorXi assignClusters(Eigen::ArrayXXd dataPoints) const {
    Eigen::VectorXi assignments = Eigen::VectorXi::Zero(dataPoints.rows());
    for (index i = 0; i < dataPoints.rows(); i++) {
      assignments(i) = assignPoint(dataPoints.row(i));
    }
    return assignments;
  }

  void computeMeans(Eigen::ArrayXXd dataPoints) {
    using namespace Eigen;
    for (index k = 0; k < mK; k++) {
      if (mEmpty[k])
        continue;
      std::vector<index> kAssignment;
      for (index i = 0; i < mAssignments.size(); i++) {
        if (mAssignments(i) == k)
          kAssignment.push_back(i);
      }
      if (kAssignment.size() == 0) {
        std::cout << "Warning: empty cluster" << std::endl;
        mEmpty[k] = true;
        return;
      }
      ArrayXXd clusterPoints = ArrayXXd::Zero(kAssignment.size(), mDims);
      for (index i = 0; i < kAssignment.size(); i++) {
        clusterPoints.row(i) = dataPoints.row(kAssignment[i]);
      }
      ArrayXd mean = clusterPoints.colwise().mean();
      mMeans.row(k) = mean;
    }
  }

  bool changed(Eigen::VectorXi newAssignments) const {
    auto dif = (newAssignments - mAssignments).cwiseAbs().sum();
    return dif > 0;
  }

  index mK{0};
  index mDims{0};
  Eigen::ArrayXXd mMeans;
  std::vector<bool> mEmpty;
  Eigen::VectorXi mAssignments;
  bool mTrained{false};
};
} // namespace algorithm
} // namespace fluid
