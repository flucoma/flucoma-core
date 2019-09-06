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
  void init(int dims, const FluidDataset<std::string, double, double, 1> &source)
  {
    mTree = KDTree<double>(source);
  }

  double predict(RealVectorView point, int k){
    using namespace std;
    auto nearest = mTree.kNearest(point, k);
    double prediction = 0;
    double weight = 1.0/k;
    for(int i = 0; i < k; i++){
      prediction += weight * nearest.getTargets()(i);
    }
    return prediction;
  }

private:
  KDTree<double> mTree{0};
};
} // namespace algorithm
} // namespace fluid
