#pragma once

#include "KDTree.hpp"
#include "algorithms/util/FluidEigenMappings.hpp"
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

  std::string predict(KDTree tree, RealVectorView point, LabelSet labels, index k) const{
    using namespace std;
    unordered_map<string, index> labelsMap;
    auto nearest = tree.kNearest(point, k);
    string prediction;
    index count = 0;

    for(index i = 0; i < k; i++){
      auto id = nearest.getIds()(i);
      FluidTensor<string, 1> labelT(1);
      labels.get(id, labelT);
      string label = labelT(0);
      auto pos = labelsMap.find(label);
      index kCount = 1;
      if (pos == labelsMap.end())labelsMap.insert({label, 1});
      else{
        kCount = pos->second;
      }
      if (kCount > count || count == 0){
        prediction = label;
        count = kCount;
      }
    }
    return prediction;
  }
};
} // namespace algorithm
} // namespace fluid
