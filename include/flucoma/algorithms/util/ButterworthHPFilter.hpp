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

#include "AlgorithmUtils.hpp"
#include <cassert>
#include <cmath>

namespace fluid {
namespace algorithm {

class ButterworthHPFilter
{
public:
  void init(double cutoff)
  { // as fraction of sample rate
    using namespace std;
    double c = tan(pi * cutoff);
    mB0 = 1.0 / (1.0 + sqrtTwo * c + pow(c, 2.0));
    mB1 = -2.0 * mB0;
    mB2 = mB0;
    mA0 = 2.0 * mB0 * (pow(c, 2.0) - 1.0);
    mA1 = mB0 * (1.0 - sqrtTwo * c + pow(c, 2.0));
    mXnz1 = 0;
    mXnz2 = 0;
    mYnz1 = 0;
    mYnz2 = 0;
  }

  double processSample(double x)
  {
    double y = mB0 * x + mB1 * mXnz1 + mB2 * mXnz2 - mA0 * mYnz1 - mA1 * mYnz2;
    mXnz2 = mXnz1;
    mXnz1 = x;
    mYnz2 = mYnz1;
    mYnz1 = y;
    return y;
  }

private:
  double mB0{0.0}, mB1{0.0}, mB2{0.0};
  double mA0{0.0}, mA1{0.0};
  double mXnz1{0.0};
  double mXnz2{0.0};
  double mYnz1{0.0};
  double mYnz2{0.0};
};
} // namespace algorithm
} // namespace fluid
