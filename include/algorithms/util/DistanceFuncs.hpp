/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright 2017-2019 University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/

#pragma once

#include "AlgorithmUtils.hpp"
#include <Eigen/Core>
#include <cassert>
#include <cmath>
#include <map>

namespace fluid {
namespace algorithm {

class DistanceFuncs
{

public:
  enum class Distance {
    kManhattan,
    kEuclidean,
    kSqEuclidean,
    kMax,
    kMin,
    kKL,
    kCosine
  };

  using ArrayXcd = Eigen::ArrayXcd;
  using ArrayXd = Eigen::ArrayXd;
  using MatrixXd = Eigen::MatrixXd;
  using DistanceFuncsMap =
      std::map<Distance, std::function<double(ArrayXd, ArrayXd)>>;

  static DistanceFuncsMap& map()
  {
    static DistanceFuncsMap _funcs = {
        {Distance::kManhattan,
         [](ArrayXd x, ArrayXd y) { return (x - y).abs().sum(); }},
        {Distance::kEuclidean,
         [](ArrayXd x, ArrayXd y) {
           return std::sqrt((x - y).square().sum());
         }},
        {Distance::kSqEuclidean,
         [](ArrayXd x, ArrayXd y) { return (x - y).square().sum(); }},
        {Distance::kMax,
         [](ArrayXd x, ArrayXd y) { return (x - y).abs().maxCoeff(); }},
        {Distance::kMin,
         [](ArrayXd x, ArrayXd y) { return (x - y).abs().minCoeff(); }},
        {Distance::kKL,
         [](ArrayXd x, ArrayXd y) {
           auto   logX = x.max(epsilon).log(), logY = y.max(epsilon).log();
           double d1 = (x * (logX - logY)).sum();
           double d2 = (y * (logY - logX)).sum();
           return d1 + d2;
         }},
        {Distance::kCosine, [](ArrayXd x, ArrayXd y) {
           double norm = x.matrix().norm() * y.matrix().norm();
           double dot = x.matrix().dot(y.matrix());
           return dot / norm;
         }}};
    return _funcs;
  }
};

Eigen::MatrixXd DistanceMatrix(Eigen::Ref<Eigen::MatrixXd> X, index distance)
{
  auto            dist = static_cast<DistanceFuncs::Distance>(distance);
  Eigen::MatrixXd D = Eigen::MatrixXd::Zero(X.rows(), X.rows());
  for (index i = 0; i < X.rows(); i++)
  {
    for (index j = 0; j < X.rows(); j++)
    {
      D(i, j) = DistanceFuncs::map()[dist](X.row(i).array(), X.row(j).array());
    }
  }
  return D;
}

template <typename Derived>
Eigen::MatrixXd DistanceMatrix(const Eigen::PlainObjectBase<Derived>& X,
                               const Eigen::PlainObjectBase<Derived>& Y,
                               index                                  distance)
{
  auto            dist = static_cast<DistanceFuncs::Distance>(distance);
  Eigen::MatrixXd D = Eigen::MatrixXd::Zero(X.rows(), Y.rows());
  for (index i = 0; i < X.rows(); i++)
  {
    for (index j = 0; j < Y.rows(); j++)
    {
      D(i, j) = DistanceFuncs::map()[dist](X.row(i).array(), Y.row(j).array());
    }
  }
  return D;
}


} // namespace algorithm
} // namespace fluid
