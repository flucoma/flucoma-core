#pragma once

#include "algorithms/util/FluidEigenMappings.hpp"
#include "data/FluidDataset.hpp"
#include "data/FluidTensor.hpp"
#include "data/TensorTypes.hpp"
#include <Eigen/Core>
#include <queue>
#include <string>

namespace fluid {
namespace algorithm {

class KMeans {

public:
  void init(int k, int dims) {
    mK = k;
    mDims = dims;
  }

  void train(const FluidDataset<std::string, double, std::string, 1> &dataset, int maxIter,
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
      mAssignments = ((0.5 + (0.5 * ArrayXf::Random(dataPoints.rows()))) * (mK - 1))
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
      std::cout<<maxIter<<std::endl;
    }
  }

  int vq(RealVectorView point) {
    assert(point.size() == mDims);
    return assignPoint(_impl::asEigen<Eigen::Array>(point));
  }

  void getMeans(RealMatrixView out){
    out = _impl::asFluid(mMeans);
  }

  void setMeans(RealMatrixView means){
    mMeans = _impl::asEigen<Eigen::Array>(means);
  }

  int getDims(){
    return mDims;
  }
  int getK(){
    return mK;
  }

private:
  double distance(Eigen::ArrayXd v1, Eigen::ArrayXd v2) {
    return (v1 - v2).matrix().norm();
  }

  int assignPoint(Eigen::ArrayXd point) {
    double minDistance = std::numeric_limits<double>::infinity();
    int minK;
    for (int k = 0; k < mK; k++) {
      double dist = distance(point, mMeans.row(k));
      if (dist < minDistance) {
        minK = k;
        minDistance = dist;
      }
    }
    return minK;
  }

  Eigen::VectorXi assignClusters(Eigen::ArrayXXd dataPoints) {
    Eigen::VectorXi assignments = Eigen::VectorXi::Zero(dataPoints.rows());
    for (int i = 0; i < dataPoints.rows(); i++) {
      assignments(i) = assignPoint(dataPoints.row(i));
    }
    return assignments;
  }

  void computeMeans(Eigen::ArrayXXd dataPoints) {
    using namespace Eigen;
    for (int k = 0; k < mK; k++) {
      if(mEmpty[k]) continue;
      std::vector<int> kAssignment;
      for (int i = 0; i < mAssignments.size(); i++) {
        if (mAssignments(i) == k)
          kAssignment.push_back(i);
      }
      if (kAssignment.size() == 0)
      {
        std::cout<<"Warning: empty cluster"<<std::endl;
        mEmpty[k] = true;
        return;
      }
      ArrayXXd clusterPoints = ArrayXXd::Zero(kAssignment.size(), mDims);
      for (int i = 0; i < kAssignment.size(); i++) {
        clusterPoints.row(i) = dataPoints.row(kAssignment[i]);
      }
      ArrayXd mean = clusterPoints.colwise().mean();
      mMeans.row(k) = mean;
    }
  }

  bool changed(Eigen::VectorXi newAssignments) {
    auto dif = (newAssignments - mAssignments).cwiseAbs().sum();
    return dif > 0;
  }

  int mK{0};
  int mDims{0};
  Eigen::ArrayXXd mMeans;
  std::vector<bool> mEmpty;
  Eigen::VectorXi mAssignments;
};
} // namespace algorithm
} // namespace fluid
