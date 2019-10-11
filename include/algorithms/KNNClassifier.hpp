#pragma once

#include "KDTree.hpp"
#include "algorithms/util/FluidEigenMappings.hpp"
#include "data/FluidDataset.hpp"
#include "data/FluidTensor.hpp"
#include "data/TensorTypes.hpp"
#include <string>

namespace fluid {
namespace algorithm {

class KNNClassifier {

public:
  using LabelSet = FluidDataset<std::string, double, std::string, 1>;

  std::string predict(KDTree<std::string> tree, RealVectorView point, LabelSet labels, int k){
    using namespace std;
    unordered_map<string, int> labelsMap;
    auto nearest = tree.kNearest(point, k);
    string prediction;
    int count = 0;

    for(int i = 0; i < k; i++){
      auto id = nearest.getIds()(i);
      auto target = labels.getTarget(id);
      //auto target = nearest.getTargets()(i);
      auto pos = labelsMap.find(target);
      int kCount = 1;
      if (pos == labelsMap.end())labelsMap.insert({target, 1});
      else{
        kCount = pos->second;
      }
      if (kCount > count || count == 0){
        prediction = target;
        count = kCount;
      }
    }
    return prediction;
  }
};
} // namespace algorithm
} // namespace fluid
