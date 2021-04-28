#pragma once

#include "KDTree.hpp"
#include "algorithms/util/FluidEigenMappings.hpp"
#include "algorithms/util/AlgorithmUtils.hpp"
#include "data/FluidDataSet.hpp"
#include "data/FluidTensor.hpp"
#include "data/TensorTypes.hpp"
#include "data/FluidIndex.hpp"
#include <string>

namespace fluid {
namespace algorithm {

class KNNClassifier {

public:
  using LabelSet = FluidDataSet<std::string, std::string, 1>;

  std::string predict(
    KDTree tree, RealVectorView point,
    LabelSet labels, index k, bool weighted) const
  {
    using namespace std;
    unordered_map<string, double> labelsMap;
    auto nearest = tree.kNearest(point, k);
    auto ids = nearest.getIds();
    auto distances = nearest.getData();

    double uniformWeight = 1.0 / k;
    std::vector<double> weights;
    double sum = 0;
    if(weighted){
      weights = std::vector<double>(asUnsigned(k), 0);
      bool binaryWeights = false;
      for(index i = 0; i < k; i++){
        if (distances(i,0) < epsilon) {
          binaryWeights = true;
          weights[asUnsigned(i)] = 1;
        }
        else sum += (1.0 / distances(i,0));
      }
      if (!binaryWeights){
        for(index i = 0; i < k; i++){
          weights[asUnsigned(i)] = (1.0 / distances(i,0)) / sum;
        }
      }
    } else {
      weights = std::vector<double>(asUnsigned(k), uniformWeight);
    }

    string prediction;
    FluidTensor<string, 1> tmp(1);
    double maxWeight = 0;
    for(index i = 0; i < k; i++){
      labels.get(ids(i), tmp);
      string label = tmp(0);
      auto pos = labelsMap.find(label);
      if (pos == labelsMap.end())labelsMap[label] = weights[asUnsigned(i)];
      else labelsMap[label] += weights[asUnsigned(i)];
      if(labelsMap[label] > maxWeight){
        maxWeight = labelsMap[label] ;
        prediction = label;
      }
    }
    return prediction;
  }
};
} // namespace algorithm
} // namespace fluid
