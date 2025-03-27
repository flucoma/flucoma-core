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

#include "../../data/FluidIndex.hpp"
#include <Eigen/Core>
#include <cmath>

namespace fluid {
namespace algorithm {

class Stats
{
public:
  using ArrayXd = Eigen::ArrayXd;
  ArrayXd process(Eigen::Ref<ArrayXd> input, double low, double mid,
                  double high)
  {
    using namespace std;
    index   length = input.size();
    ArrayXd out = ArrayXd::Zero(7);
    double  mean = input.mean();
    double  stdev = sqrt((input - mean).square().mean());
    double skewness = ((input - mean) / (stdev == 0 ? 1 : stdev)).cube().mean();
    double kurtosis = ((input - mean) / (stdev == 0 ? 1 : stdev)).pow(4).mean();
    ArrayXd sorted = input;
    sort(sorted.data(), sorted.data() + length);
    double lowVal = sorted(lrint(low * (length - 1)));
    double midVal = sorted(lrint(mid * (length - 1)));
    double highVal = sorted(lrint(high * (length - 1)));
    out << mean, stdev, skewness, kurtosis, lowVal, midVal, highVal;
    return out;
  }
};
} // namespace algorithm
} // namespace fluid
