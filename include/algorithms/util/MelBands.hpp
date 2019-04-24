#pragma once

#include "../util/FluidEigenMappings.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cassert>
#include <cmath>

namespace fluid {
namespace algorithm {

using _impl::asEigen;
using _impl::asFluid;
using Eigen::ArrayXd;
using Eigen::ArrayXXd;
using Eigen::MatrixXd;
using Eigen::Array;

class MelBands {
public:
  /*static inline double mel2hz(double x) {
      return 700.0 * (exp(x / 1127.01048) - 1.0);
    }*/

  static inline double hz2mel(double x) {
    return 1127.01048 * log(x / 700.0 + 1.0);
  }

  void init(double lo, double hi, int nBands, int nBins,
            double sampleRate) {
    assert(hi > lo);
    assert(nBands > 1);
    mLo = lo;
    mHi = hi;
    mBands = nBands;
    mSampleRate = sampleRate;
    mBins = nBins;
    ArrayXd melFreqs = ArrayXd::LinSpaced(mBands + 2, hz2mel(lo), hz2mel(hi));
    melFreqs = 700.0 * ((melFreqs / 1127.01048).exp() - 1.0);
    mFilters = MatrixXd::Zero(mBands, mBins);
    ArrayXd fftFreqs = ArrayXd::LinSpaced(mBins, 0, mSampleRate / 2.0);
    ArrayXd melD =
        (melFreqs.segment(0, mBands + 1) - melFreqs.segment(1, mBands + 1))
            .abs();
    ArrayXXd ramps = melFreqs.replicate(1, mBins);
    ramps.rowwise() -= fftFreqs.transpose();
    for (int i = 0; i < mBands; i++) {
      ArrayXd lower = -ramps.row(i) / melD(i);
      ArrayXd upper = ramps.row(i + 2) / melD(i + 1);
      mFilters.row(i) = lower.min(upper).max(0);
    }
    ArrayXd enorm =
        2.0 / (melFreqs.segment(2, mBands) - melFreqs.segment(0, mBands));
    mFilters = (mFilters.array().colwise() *= enorm).matrix();
    // mOutputBuffer = ArrayXd::Zero(mBands);
  }

  // Eigen::Ref<ArrayXd> processFrame(Eigen::Ref<const ArrayXd> input) {
  void processFrame(const RealVector in, RealVectorView out) {
    assert(in.size() == mBins);
    ArrayXd frame = asEigen<Array>(in);
    // mOutputBuffer = mFilters * input.matrix();
    // return mOutputBuffer;
    ArrayXd result = (mFilters * frame.matrix()).array();
    out = asFluid(result);
  }

  double mLo;
  double mHi;
  int mBins;
  int mBands;
  double mSampleRate;
  MatrixXd mFilters;

private:
  // ArrayXd mOutputBuffer;
};
}; // namespace algorithm
}; // namespace fluid
