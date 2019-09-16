#pragma once

#include "KDTree.hpp"
#include "algorithms/util/FluidEigenMappings.hpp"
#include "data/FluidDataset.hpp"
#include "data/FluidTensor.hpp"
#include "data/TensorTypes.hpp"
#include <string>


namespace fluid {
namespace algorithm {

class KNNRegressor {

public:

  double predict(KDTree<double> tree, RealVectorView point, int k){
    using namespace std;
    auto nearest = tree.kNearest(point, k);
    double prediction = 0;
    double weight = 1.0/k;
    for(int i = 0; i < k; i++){
      prediction += weight * nearest.getTargets()(i);
    }
    return prediction;
  }

};
} // namespace algorithm
} // namespace fluid
