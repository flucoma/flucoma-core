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

  void predict(KDTree const& tree, DataSet const& targets,
                 RealVectorView input, RealVectorView output,
                 index k, bool weighted,
                 Allocator& alloc = FluidDefaultAllocator()) const
  {
    using namespace std;
    using _impl::asEigen;
    using Eigen::Array;

    auto [distances, ids] = tree.kNearest(input, k, 0, alloc);

    ScopedEigenMap<Eigen::VectorXd> weights(k, alloc);
    weights.setConstant(weighted ? 0 : (1.0 / k));

    if (weighted)
    {
      auto distanceArray =
          Eigen::Map<Eigen::ArrayXd>(distances.data(), distances.size());

      if ((distanceArray < epsilon).any())
      {
        weights = (distanceArray < epsilon).select(1.0, weights);
      }
      else
      {
        double sum = (1.0 / distanceArray).sum();
        weights = (1.0 / distanceArray) / sum;
      }
    }
      
      output.fill(0);
      for (size_t i = 0; i < asUnsigned(k); i++)
      {
        Eigen::ArrayXd point = asEigen<Array>(targets.get(*ids[i]));
        asEigen<Array>(output) += (weights[i] * point);
      }
  }
};
} // namespace algorithm
} // namespace fluid

