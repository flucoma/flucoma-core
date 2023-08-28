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

// divide-and-conquer algorithm using the DTW algorithm
// and various constraints to reduce complexity
// down from O(N*M) (boooo) to O(N) (yayyy)
// props to the absolute units Stan Salvador and Philip Chan
// over at https://cs.fit.edu/~pkc/papers/tdm04.pdf

class FastDTW
{
  using MatrixXd = Eigen::MatrixXd;
  using VectorXd = Eigen::VectorXd;
  using ArrayXd = Eigen::ArrayXd;

public:
  explicit FastDTW() = default;
  ~FastDTW() = default;

  // functions so the DataClient<FastDTW> doesnt have freak out
  void init() { mInitialised = true; }

  void clear() const {}

  index size() const { return mConstraintSize; }
  index dims() const { return mConstraintSize; }

  bool initialized() const { return mInitialised; }

  void process() const {}

private:
  bool  mInitialised{false};
  index mConstraintSize{8};
};

} // namespace algorithm
} // namespace fluid