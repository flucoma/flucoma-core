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

namespace fluid {
namespace algorithm {

class SlideUDFilter
{
public:
  void updateCoeffs(double rampUpTime, double rampDownTime)
  {
    mBUp = 1.0 / rampUpTime;
    mBDown = 1.0 / rampDownTime;
  }

  void init(double x0Val) { y0 = x0Val; }

  double processSample(double x)
  {
    double y;
    if (x > y0) { y = y0 + (mBUp * (x - y0)); }
    else
    {
      y = y0 + (mBDown * (x - y0));
    }
    y0 = y;
    return y;
  }

private:
  double mBUp{0.0}, mBDown{0.0};
  double y0{0.0};
};
} // namespace algorithm
} // namespace fluid
