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

#include "FluidEigenMappings.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/FluidTensor.hpp"
#include <Eigen/Core>
#include <limits>

namespace fluid {
namespace algorithm {

class OutlierDetection
{

  using ArrayXd = Eigen::ArrayXd;
  using ArrayXi = Eigen::ArrayXi;

public:
  void process(Eigen::Ref<const ArrayXd> input, Eigen::Ref<Eigen::ArrayXi> mask,
               double k)
  {
    index   length = input.size();
    ArrayXidx perm = ArrayXidx::LinSpaced(length, 0, length - 1);
    std::sort(perm.data(), perm.data() + length,
              [&](index i, index j) { return input(i) < input(j); });
    index  q1 = lrint(0.25 * (length - 1));
    index  q3 = lrint(0.75 * (length - 1));
    double margin = k * (input(perm(q3)) - input(perm(q1)));
    double lowerBound = input(perm(q1)) - margin;
    double upperBound = input(perm(q3)) + margin;
    for (index i = 0; input(perm(i)) < lowerBound && i <= q1; i++)
    { mask(perm(i)) = 0; }
    for (index i = length - 1; input(perm(i)) > upperBound && i >= q3; i--)
    { mask(perm(i)) = 0; }
  }
};
} // namespace algorithm
} // namespace fluid
