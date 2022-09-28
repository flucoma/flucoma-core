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
#include "../../data/FluidMemory.hpp"
#include <string>

namespace fluid {
namespace algorithm {

class KNNRegressor
{

public:
  using DataSet = FluidDataSet<std::string, double, 1>;

  double predict(KDTree const& tree, DataSet const& targets,
                 RealVectorView point, index k, bool weighted,
                 Allocator& alloc = FluidDefaultAllocator()) const
  {
    using namespace std;
    double prediction = 0;
    auto [distances, ids] = tree.kNearest(point, k, 0, alloc);
    double             uniformWeight = 1.0 / k;
    rt::vector<double> weights(asUnsigned(k), weighted ? 0 : uniformWeight,
                               alloc);
    double             sum = 0;
    if (weighted)
    {
      bool binaryWeights = false;
      for (size_t i = 0; i < asUnsigned(k); i++)
      {
        if (distances[i] < epsilon)
        {
          binaryWeights = true;
          weights[i] = 1;
        }
        else
          sum += (1.0 / distances[i]);
      }
      if (!binaryWeights)
      {
        for (size_t i = 0; i < asUnsigned(k); i++)
        {
          weights[i] = (1.0 / distances[i]) / sum;
        }
      }
    }

    for (size_t i = 0; i < asUnsigned(k); i++)
    {
      auto point = targets.get(*ids[i]);
      prediction += (weights[i] * point(0));
    }
    return prediction;
  }
};
} // namespace algorithm
} // namespace fluid
