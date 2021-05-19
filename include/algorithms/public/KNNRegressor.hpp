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

#include "KDTree.hpp"
#include "../util/AlgorithmUtils.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../../data/FluidDataSet.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/FluidTensor.hpp"
#include "../../data/TensorTypes.hpp"
#include <string>

namespace fluid {
namespace algorithm {

class KNNRegressor
{

public:
  using DataSet = FluidDataSet<std::string, double, 1>;

  double predict(KDTree tree, DataSet targets, RealVectorView point, index k,
                 bool weighted) const
  {
    using namespace std;
    auto                nearest = tree.kNearest(point, k);
    double              prediction = 0;
    auto                ids = nearest.getIds();
    auto                distances = nearest.getData();
    double              uniformWeight = 1.0 / k;
    std::vector<double> weights;
    double              sum = 0;
    if (weighted)
    {
      weights = std::vector<double>(asUnsigned(k), 0);
      bool binaryWeights = false;
      for (index i = 0; i < k; i++)
      {
        if (distances(i, 0) < epsilon)
        {
          binaryWeights = true;
          weights[asUnsigned(i)] = 1;
        }
        else
          sum += (1.0 / distances(i, 0));
      }
      if (!binaryWeights)
      {
        for (index i = 0; i < k; i++)
        { weights[asUnsigned(i)] = (1.0 / distances(i, 0)) / sum; }
      }
    }
    else
    {
      weights = std::vector<double>(asUnsigned(k), uniformWeight);
    }
    for (index i = 0; i < k; i++)
    {
      auto point = FluidTensor<double, 1>(1);
      targets.get(ids(i), point);
      prediction += (weights[asUnsigned(i)] * point(0));
    }
    return prediction;
  }
};
} // namespace algorithm
} // namespace fluid
