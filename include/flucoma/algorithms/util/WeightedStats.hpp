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
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>
#include <cassert>
#include <cmath>

namespace fluid {
namespace algorithm {

class WeightedStats
{
public:
  Eigen::ArrayXd process(Eigen::Ref<Eigen::ArrayXd> input,
                         Eigen::Ref<Eigen::ArrayXd> weights, double low,
                         double mid, double high)
  {
    using namespace Eigen;
    using namespace std;
    index   length = input.size();
    ArrayXd out = ArrayXd::Zero(7);
    double  mean = (weights * input).sum();
    double  stdev = sqrt((weights * (input - mean).square()).sum());
    double  skewness =
        (weights * ((input - mean) / (stdev == 0 ? 1 : stdev)).cube()).sum();
    double kurtosis =
        (weights * ((input - mean) / (stdev == 0 ? 1 : stdev)).pow(4)).sum();
    ArrayXd sorted = input;
    ArrayXidx perm = ArrayXidx::LinSpaced(length, 0, length - 1);
    std::sort(perm.data(), perm.data() + length,
              [&](index i, index j) { return input(i) < input(j); });
    index  level = 0;
    double lowVal{input(perm(0))};
    double midVal{input(perm(lrint(length - 1) / 2))};
    double hiVal{input(perm(lrint(length - 1)))};
    double acc = weights(perm(0)), prevAcc = 0;
    for (index i = 1; i < length; i++)
    {
      acc += weights(perm(i));
      if (level == 0 && acc >= low)
      {
        lowVal = abs(prevAcc - low) <= abs(acc - low) ? input(perm(i - 1))
                                                      : input(perm(i));
        level = 1;
      }
      if (level == 1 && acc >= mid)
      {
        midVal = abs(prevAcc - mid) < abs(acc - mid) ? input(perm(i - 1))
                                                     : input(perm(i));
        level = 2;
      }
      if (level == 2 && acc >= high)
      {
        hiVal = abs(prevAcc - high) < abs(acc - high) ? input(perm(i - 1))
                                                      : input(perm(i));
        break;
      }
      prevAcc = acc;
    }
    out << mean, stdev, skewness, kurtosis, lowVal, midVal, hiVal;
    return out;
  }
};
} // namespace algorithm
} // namespace fluid
