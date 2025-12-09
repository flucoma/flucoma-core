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
#include <cassert>
#include <queue>
#include <random>
#include <string>

namespace fluid {
namespace algorithm {

namespace _impl::kmeans_init {

/// @brief Initialize means based on randomly assigning each point to a cluster
/// @param input input data
/// @param k number of clusters
/// @return a 2D Eigen array of means
auto randomPartition(const Eigen::MatrixXd& input, index k)
{
  // Means come from randomly assigning points and taking average
  std::random_device              rd;
  std::mt19937                    gen(rd());
  std::uniform_int_distribution<index> distrib(0, k - 1);
  Eigen::ArrayXXd means = Eigen::ArrayXXd::Zero(k, input.cols());
  Eigen::ArrayXi  weights(k);
  std::for_each_n(input.rowwise().begin(), input.rows(),
                  [&gen, &distrib, &means, &weights](auto row) {
                    index label = distrib(gen);
                    means(label, Eigen::all) += row.array();
                    weights(label)++;
                  });
  means /= weights.replicate(1, 2).cast<double>();
  return means;
}

/// @brief Initialize means by sampling `k` random points ('Forgy initialization')
/// @param input input data
/// @param k number of clusters
/// @return 2D Eigen expression of sampled input points
auto randomPoints(const Eigen::MatrixXd& input, index k)
{
  // Means come from k random points 
  std::random_device              rd;
  std::mt19937                    gen(rd());
  std::uniform_int_distribution<index> distrib(0, input.rows() - 1);

  std::vector<index> rows(asUnsigned(k));
  std::generate(begin(rows), end(rows),
                [&distrib, &gen]() { return distrib(gen); });
  return input(rows, Eigen::all);
}

auto squareEuclidiean = [](Eigen::Ref<const Eigen::MatrixXd> const& a,
                          Eigen::Ref<const Eigen::MatrixXd> const& b,
                          bool squared = true) {
  double a_sqnorm = a.squaredNorm(); 
  double b_sqnorm = b.squaredNorm(); 
  Eigen::ArrayXXd result = (a * b.transpose()).array();                             
  result *= -2; 
  result += (a_sqnorm + b_sqnorm);                         
  return squared ? result: result.sqrt(); 
};

auto cosine = [](auto a, auto b){
  return 1.0 - (a * b.transpose()).array();  
}; 

/// @brief initilaize means using markov chain montecarlo approximation of Kmeans++ (kmc2)
/// @tparam DistanceFn function object that performs distance calculation
/// @param input 
/// @param k 
/// @param distance 
/// @return 
template<class DistanceFn>
auto akmc2(Eigen::MatrixXd const& input, index k, DistanceFn distance)
{
  std::random_device rd;
  std::mt19937       gen(rd());
  Eigen::MatrixXd centres(k, input.cols()); 
  
  // First mean sampled at random from input 
  const index        centre0 =
      std::uniform_int_distribution<index>(0, input.rows() - 1)(gen);
  centres.row(0) = input.row(centre0);

  Eigen::ArrayXd q = distance(input, centres.row(0)).pow(2);   
  q /= (2 * q.sum() + 2 * q.rows());
  std::discrete_distribution  proposalDistribution(q.begin(), q.end());

  index    chainLength = 200;
  auto candidateIdx = std::vector<index>(asUnsigned(chainLength)); 
  Eigen::VectorXd candidateProbs(chainLength); 
  std::uniform_real_distribution<double> uniform;

  std::generate_n(centres.rowwise().begin() + 1, k - 1, [&, i = 0]() mutable {
    std::generate(
        candidateIdx.begin(), candidateIdx.end(),
        [&gen, &proposalDistribution]() { return proposalDistribution(gen); });

    Eigen::VectorXd proposalProbabilities = q(candidateIdx);

    // changes size every iteration
    Eigen::ArrayXXd dist = distance(input(candidateIdx, Eigen::all),
                                    centres(Eigen::seq(0, i++), Eigen::all));
    candidateProbs = dist.rowwise().minCoeff() / q(candidateIdx);

    auto start = candidateProbs.begin();
    auto current = start;
    for (auto it = start; it != candidateProbs.end(); ++it)
    {
      if (*current == 0.0 || *it / *current > uniform(gen)) current = it;
    }
    return input.row(candidateIdx[asUnsigned(std::distance(start, current))]);
  });
  return centres; 
}
} //_impl::kmeans_init

class KMeans
{

public:
  enum class InitMethod {randomPartion, randomPoint, randomSampling}; 
  
  void clear()
  {
    mMeans.setZero();
    mAssignments.resize(0);
    mTrained = false;
  }

  bool initialized() const { return mTrained; }

  void train(const FluidDataSet<std::string, double, 1>& dataset, index k,
             index maxIter, InitMethod init)
  {
    using namespace Eigen;
    using namespace _impl;
    assert(!mTrained || (dataset.pointSize() == mDims && mK == k));
    auto dataPoints = asEigen<Array>(dataset.getData());
    
    if(!mTrained)
    {
      mK = k;
      mDims = dataset.pointSize();

      using namespace _impl::kmeans_init; 
      switch(init)
      {
        case InitMethod::randomSampling: 
        { 
          mMeans = akmc2(dataPoints, mK, squareEuclidiean); 
          break; 
        }
        case InitMethod::randomPoint: 
        {
            mMeans = randomPoints(dataPoints, mK); 
            break; 
        }
        default: mMeans = randomPartition(dataPoints, mK); 
      }

      mEmpty = std::vector<bool>(asUnsigned(mK), false);
    }

    Eigen::VectorXi assignments(dataPoints.rows()); 
    while (maxIter-- > 0)
    {      
      assignments = assignClusters(dataPoints);
      if (!changed(assignments)) { break; }
      else
      {
        mAssignments = assignments;
      }
      computeMeans(dataPoints);
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
    if (mAssignments.rows() == 0) return true; 
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
