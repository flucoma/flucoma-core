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
#include "../../data/FluidIndex.hpp"
#include <cmath>

namespace fluid {
namespace algorithm {

class KWeightingFilter
{
public:
  void init(double sampleRate)
  {
    // from https://github.com/jiixyj/libebur128/blob/master/ebur128/ebur128.c

    // Shelving filter
    using namespace std;

    double f0 = 1681.974450955533;
    double G = 3.999843853973347;
    double Q = 0.7071752369554196;

    double K = tan(pi * f0 / sampleRate);
    double Vh = pow(10.0, G / 20.0);
    double Vb = pow(Vh, 0.4996667741545416);

    double shelvB[3] = {0.0, 0.0, 0.0};
    double shelvA[3] = {1.0, 0.0, 0.0};

    double a0 = 1.0 + K / Q + K * K;
    shelvB[0] = (Vh + Vb * K / Q + K * K) / a0;
    shelvB[1] = 2.0 * (K * K - Vh) / a0;
    shelvB[2] = (Vh - Vb * K / Q + K * K) / a0;
    shelvA[1] = 2.0 * (K * K - 1.0) / a0;
    shelvA[2] = (1.0 - K / Q + K * K) / a0;

    // Hi-pass filter
    f0 = 38.13547087602444;
    Q = 0.5003270373238773;
    K = tan(pi * f0 / sampleRate);

    double hiB[3] = {1.0, -2.0, 1.0};
    double hiA[3] = {1.0, 0.0, 0.0};

    hiA[1] = 2.0 * (K * K - 1.0) / (1.0 + K / Q + K * K);
    hiA[2] = (1.0 - K / Q + K * K) / (1.0 + K / Q + K * K);

    // Combined
    mB[0] = shelvB[0] * hiB[0];
    mB[1] = shelvB[0] * hiB[1] + shelvB[1] * hiB[0];
    mB[2] = shelvB[0] * hiB[2] + shelvB[1] * hiB[1] + shelvB[2] * hiB[0];
    mB[3] = shelvB[1] * hiB[2] + shelvB[2] * hiB[1];
    mB[4] = shelvB[2] * hiB[2];

    mA[0] = shelvA[0] * hiA[0];
    mA[1] = shelvA[0] * hiA[1] + shelvA[1] * hiA[0];
    mA[2] = shelvA[0] * hiA[2] + shelvA[1] * hiA[1] + shelvA[2] * hiA[0];
    mA[3] = shelvA[1] * hiA[2] + shelvA[2] * hiA[1];
    mA[4] = shelvA[2] * hiA[2];
    for (index i = 0; i < 5; i++)
    {
      mX[i] = 0;
      mY[i] = 0;
    }
  }

  double processSample(double x)
  {
    double y = 0;
    mX[0] = x;
    for (index i = 1; i < 5; i++) y -= mA[i] * mY[i - 1];
    for (index i = 0; i < 5; i++) y += mB[i] * mX[i];

    for (index i = 4; i > 0; i--)
    {
      mX[i] = mX[i - 1];
      mY[i] = mY[i - 1];
    }
    mY[0] = y;
    return y;
  }

private:
  double mA[5], mB[5], mX[5], mY[5];
};

} // namespace algorithm
} // namespace fluid
