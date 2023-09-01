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

#include "../util/FluidEigenMappings.hpp"
#include "../../data/FluidDataSet.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/FluidMemory.hpp"
#include "../../data/FluidTensor.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>
#include <random>

namespace fluid {
namespace algorithm {

template <class Cell>
class Recur
{
  using Cell::StateType;

public:
  void forward();

  Cell      mCell;
  StateType state;

private:
  bool mInitialized;
  bool mTrained;

  // rt vector of cells (each have ptr to params)
  // pointer so Recur can be default constructible
  rt::vector<Cell, Allocator> mNodes;
  std::shared_ptr<CellParams> mParams;
};

} // namespace algorithm
} // namespace fluid