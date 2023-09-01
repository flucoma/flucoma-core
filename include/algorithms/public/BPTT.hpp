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

#include "LSTMCell.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../../data/FluidDataSeries.hpp"
#include "../../data/FluidDataSet.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/FluidTensor.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>
#include <limits>
#include <random>

namespace fluid {
namespace algorithm {

template <typename Model>
class BPTT
{
  explicit BPTT() = default;
  ~BPTT() = default;

  double trainClassifier(Model&                                 model,
                         const std::vector<InputRealMatrixView> in,
                         RealMatrixView out, double learningRate);
};
} // namespace algorithm
} // namespace fluid