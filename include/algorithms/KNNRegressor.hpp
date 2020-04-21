#pragma once

#include "KDTree.hpp"
#include "algorithms/util/FluidEigenMappings.hpp"
#include "data/FluidDataSet.hpp"
#include "data/FluidTensor.hpp"
#include "data/FluidIndex.hpp"
#include "data/TensorTypes.hpp"
#include <string>


namespace fluid {
namespace algorithm {

class KNNRegressor {

public:

  using DataSet = FluidDataSet<std::string, double, 1>;
  double predict(KDTree tree, DataSet targets, RealVectorView point, index k) const{
    using namespace std;
    auto nearest = tree.kNearest(point, k);
    double prediction = 0;
    double weight = 1.0/k;
    for(index i = 0; i < k; i++){
      auto id = nearest.getIds()(i);
      auto point = FluidTensor<double, 1>(1);
      targets.get(id, point);
      prediction += weight * point(0);
    }
    return prediction;
  }

};
} // namespace algorithm
} // namespace fluid
