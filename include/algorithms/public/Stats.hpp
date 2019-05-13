#pragma once

#include "../../data/TensorTypes.hpp"
#include "../util/FluidEigenMappings.hpp"
#include <Eigen/Core>
#include <cassert>
#include <cmath>

namespace fluid {
namespace algorithm {

using _impl::asEigen;
using _impl::asFluid;
using Eigen::Array;
using Eigen::ArrayXd;
using Eigen::ArrayXXd;
using Eigen::MatrixXd;
using Eigen::Ref;

class Stats {
public:
  Stats():mNumDerivatives(0), mLow(0), mMiddle(0.5), mHigh(1){}
  void init(int numDerivatives, double low, double mid, double high) {
    assert(numDerivatives <= 2);
    mNumDerivatives = numDerivatives;
    mLow = low;
    mMiddle = mid;
    mHigh = high;
  }
int numStats(){
    return 7;
  }

  Ref<ArrayXd> computeStats(Ref<ArrayXd> input) {
    int length = input.size();
    ArrayXd out = ArrayXd::Zero(7);
    double mean = input.mean();
    double std = std::sqrt((input - mean).square().mean());
    double skewness = ((input - mean) / (std == 0 ? 1 : std)).cube().mean();
    double kurtosis = ((input - mean) / (std == 0 ? 1 : std)).pow(4).mean();
    std::sort(input.data(), input.data() + length);
    double low = input(std::round(mLow * (length - 1)));
    double mid = input(std::round(mMiddle * (length - 1)));
    double high = input(std::round(mHigh * (length - 1)));
    out << mean, std, skewness, kurtosis, low, mid, high;
    return out;
  }

  void process(const RealVectorView in, RealVectorView out) {
    using _impl::asFluid;
    using fluid::Slice;
    assert(out.size() == numStats() * (mNumDerivatives + 1));
    ArrayXd input = asEigen<Array>(in);
    int length = input.size();
    ArrayXd raw = computeStats(input);
    out(Slice(0, numStats())) = asFluid(raw);
    if (mNumDerivatives > 0) {
      ArrayXd diff1 =
          input.segment(1, length - 1) - input.segment(0, length - 1);
      ArrayXd d1 = computeStats(diff1);
      out(Slice(numStats(), numStats())) = asFluid(d1);
      if (mNumDerivatives > 1) {
        ArrayXd diff2 =
            diff1.segment(1, length - 2) - diff1.segment(0, length - 2);
        ArrayXd d2 = computeStats(diff2);
        out(Slice(2* numStats(), numStats())) = asFluid(d2);
      }
    }
  }

  int mNumDerivatives;
  double mLow;
  double mMiddle;
  double mHigh;
};
}; // namespace algorithm
}; // namespace fluid
