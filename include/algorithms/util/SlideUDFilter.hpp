#pragma once
#include <cmath>

namespace fluid {
namespace algorithm {

class SlideUDFilter {
public:
  void init(double rampUpTime, double rampDownTime) {
    mBUp = 1.0 / rampUpTime;
    mBDown = 1.0 / rampDownTime;
  }

  double processSample(double x) {
    double y;
    if (x > x0)
      y = y0 + mBUp * (x - y0);
    else
      y = y0 + mBDown * (x - y0);
    x0 = x;
    y0 = y;
    return y;
  }

private:
  double mBUp, mBDown;
  double x0, y0;
};
}; // namespace algorithm
}; // namespace fluid
