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
  /*void init(int dims, const FluidDataset<std::string, double, std::string, 1> &source)
  {
    mTree = KDTree<std::string>(source);
  }*/


  std::string predict(KDTree<std::string> tree, RealVectorView point, int k){
    using namespace std;
    unordered_map<string, int> labels;
    auto nearest = tree.kNearest(point, k);
    nearest.print();
    string prediction;
    int count = 0;

    for(int i = 0; i < k; i++){
      auto target = nearest.getTargets()(i);
      auto pos = labels.find(target);
      int kCount = 1;
      if (pos == labels.end())labels.insert({target, 1});
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
private:
  //KDTree<std::string> mTree{0};
};
} // namespace algorithm
} // namespace fluid
