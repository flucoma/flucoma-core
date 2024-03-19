/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright University of Huddersfield.
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

class KNNClassifier
{

public:
  using LabelSet = FluidDataSet<std::string, std::string, 1>;

  std::string const& predict(KDTree const& tree, RealVectorView point,
                             LabelSet const& labels, index k, bool weighted,
                             Allocator& alloc = FluidDefaultAllocator()) const
  {
    using namespace std;
    unordered_map<const string*, double> labelsMap;
    auto [distances, ids] = tree.kNearest(point, k, 0, alloc);

    double             uniformWeight = 1.0 / k;
    rt::vector<double> weights(asUnsigned(k), weighted ? 0 : uniformWeight,
                               alloc);

    double sum = 0;
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

    const string* prediction;
    double  maxWeight = 0;
    for (size_t i = 0; i < asUnsigned(k); i++)
    {
      const string* label = labels.get(*ids[i]).data();
      assert(label && "KNNClassifier: ID not mapped to label");
      auto pos = labelsMap.find(label);
      if (pos == labelsMap.end())
        labelsMap[label] = weights[i];
      else
        labelsMap[label] += weights[i];
      if (labelsMap[label] > maxWeight)
      {
        maxWeight = labelsMap[label];
        prediction = label;
      }
    }
    return *prediction;
  }
};
} // namespace algorithm
} // namespace fluid
