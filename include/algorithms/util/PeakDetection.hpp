#pragma once

#include "../../data/FluidTensor.hpp"
#include "FluidEigenMappings.hpp"

#include <Eigen/Core>
#include <limits>

namespace fluid {
namespace algorithm {

class PeakDetection {

  using ArrayXd = Eigen::ArrayXd;
  using pairs_vector = std::vector<std::pair<double, double>>;

public:
  pairs_vector process(const Eigen::Ref<ArrayXd> &input, int numPeaks = 0,
                       double minHeight = 0, bool interpolate = true) {
    using std::make_pair;
    pairs_vector peaks;

    for (int i = 1; i < input.size() - 1; i++) {
      double current = input(i);
      double prev = input(i - 1);
      double next = input(i + 1);

      if (current > prev && current > next && current > minHeight) {
        if (interpolate) {
          double p = 0.5 * (prev - next) / (prev - 2 * current + next);
          double newIndex = i + p;
          double newVal = current - 0.25 * (prev - next) * p;
          peaks.push_back(make_pair(newIndex, newVal));
        } else {
          peaks.push_back(make_pair(static_cast<double>(i), input(i)));
        }
      }
    }
      std::sort(peaks.begin(), peaks.end(), [](auto &left, auto &right) {
      return left.second > right.second;
    });
    if (numPeaks > 0  && peaks.size() > 0) {
      return pairs_vector(peaks.begin(), peaks.begin() + numPeaks);
    } else
      return peaks;
  }
};
} // namespace algorithm
} // namespace fluid
