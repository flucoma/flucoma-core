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
#include "../../data/FluidMemory.hpp"
#include "../../data/FluidTensor.hpp"
#include <Eigen/Core>
#include <limits>

namespace fluid {
namespace algorithm {

class PeakDetection
{

  using ArrayXd = Eigen::ArrayXd;
  using pairs_vector = rt::vector<std::pair<double, double>>;

public:
  pairs_vector process(const Eigen::Ref<ArrayXd>& input, index numPeaks = 0,
                       double minHeight = 0, bool interpolate = true,
                       bool       sort = true,
                       Allocator& alloc = FluidDefaultAllocator())
  {
    using std::make_pair;
    pairs_vector peaks(asUnsigned(input.size()), alloc);
    peaks.resize(0);
    for (index i = 1; i < input.size() - 1; i++)
    {
      double current = input(i);
      double prev = input(i - 1);
      double next = input(i + 1);

      if (current > prev && current > next && current > minHeight)
      {
        if (interpolate)
        {
          double p = 0.5 * (prev - next) / (prev - 2 * current + next);
          double newIndex = i + p;
          double newVal = current - 0.25 * (prev - next) * p;
          peaks.push_back(make_pair(newIndex, newVal));
        }
        else
        {
          peaks.push_back(make_pair(static_cast<double>(i), input(i)));
        }
      }
    }
    if (sort)
    {
      std::sort(peaks.begin(), peaks.end(), [](auto& left, auto& right) {
        return left.second > right.second;
      });
    }
    if (numPeaks > 0 && peaks.size() > 0)
    {
      return pairs_vector(peaks.begin(), peaks.begin() + numPeaks, alloc);
    }
    else
      return peaks;
  }
};
} // namespace algorithm
} // namespace fluid
