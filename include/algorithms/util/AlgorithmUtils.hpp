#pragma once

#include <cmath>

namespace fluid {
namespace algorithm {
  constexpr double epsilon = std::numeric_limits<double>::epsilon();
  constexpr double pi = M_PI;
  constexpr double twoPi = 2 * M_PI;
  constexpr double silence = 6.3095734448019e-08;
  constexpr double silenceDB = -144;
}; // namespace algorithm
}; // namespace fluid
