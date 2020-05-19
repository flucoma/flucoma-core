#pragma once

#include "KDTree.hpp"
#include "algorithms/util/FluidEigenMappings.hpp"
#include "algorithms/util/AlgorithmUtils.hpp"
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

  double predict(KDTree tree, DataSet targets, RealVectorView point, index k, bool weighted) const{
    using namespace std;
    auto nearest = tree.kNearest(point, k);
    double prediction = 0;
    auto ids = nearest.getIds();
    auto distances = nearest.getData();
    double uniformWeight = 1.0 / k;
    std::vector<double> weights;
    double sum = 0;
    if(weighted){
      weights = std::vector<double>(k, 0);
      bool binaryWeights = false;
      for(index i = 0; i < k; i++){
        if (distances(i,0) < epsilon) {
          binaryWeights = true;
          weights[i] = 1;
        }
        else sum += (1.0 / distances(i,0));
      }
      if (!binaryWeights){
        for(index i = 0; i < k; i++){
          weights[i] = (1.0 / distances(i,0)) / sum;
        }
      }
    } else {
      weights = std::vector<double>(k, uniformWeight);
    }
  for(index i = 0; i < k; i++){
      auto point = FluidTensor<double, 1>(1);
      targets.get(ids(i), point);
      prediction += (weights[i] * point(0));
    }
    return prediction;
  }

};
} // namespace algorithm
} // namespace fluid
