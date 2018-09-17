#pragma once

#include "MedianFilter.hpp"
#include "data/FluidEigenMappings.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <vector>

namespace fluid {
namespace rthpss {

using Eigen::Array;
using Eigen::ArrayXd;
using Eigen::ArrayXXcd;
using Eigen::ArrayXXd;
using Eigen::Dynamic;
using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::RowMajor;
using std::vector;

using ComplexVector = FluidTensorView<std::complex<double>, 1>;
using ComplexMatrix = FluidTensorView<std::complex<double>, 2>;
using ArrayXcdMap = Map<const Array<std::complex<double>, Dynamic, RowMajor>>;
using fluid::eigenmappings::ArrayXXcdToFluid;
using fluid::medianfilter::MedianFilter;

class RTHPSS {
public:
  enum HPSSMode { kClassic, kCoupled, kAdvanced };
  RTHPSS(int nBins, int vSize, int hSize, int mode, double hThresholdX1,
         double hThresholdY1, double hThresholdX2, double hThresholdY2,
         double pThresholdX1, double pThresholdY1, double pThresholdX2,
         double pThresholdY2)
      : mBins(nBins), mVSize(vSize), mHSize(hSize), mMode(mode),
        mVMedianFilter(vSize), mHMedianFilter(hSize),
        mHThresholdX1(hThresholdX1), mHThresholdY1(hThresholdY1),
        mHThresholdX2(hThresholdX2), mHThresholdY2(hThresholdY2),
        mPThresholdX1(pThresholdX1), mPThresholdY1(pThresholdY1),
        mPThresholdX2(pThresholdX2), mPThresholdY2(pThresholdY2) {
    assert(mVSize % 2);
    assert(mHSize % 2);
    assert(0 <= mMode <= 3);
    mH = ArrayXXd::Zero(mBins, hSize);
    mV = ArrayXXd::Zero(mBins, hSize);
    mBuf = ArrayXXcd::Zero(mBins, hSize);
    mHistory = ArrayXXd::Zero(mBins, 2 * hSize);
  }

  ArrayXd makeThreshold(double x1, double y1, double x2, double y2) {
    ArrayXd threshold = ArrayXd::Ones(mBins);
    int kneeStart = floor(x1 * mBins);
    int kneeEnd = floor(x2 * mBins);
    int kneeLength = kneeEnd - kneeStart;
    threshold.segment(0, kneeStart) =
        ArrayXd::Constant(kneeStart, 10).pow(y1 / 20.0);
    threshold.segment(kneeStart, kneeLength) =
        ArrayXd::Constant(kneeLength, 10)
            .pow(ArrayXd::LinSpaced(kneeLength, y1, y2) / 20.0);
    threshold.segment(kneeEnd, mBins - kneeEnd) =
        ArrayXd::Constant(mBins - kneeEnd, 10).pow(y2 / 20.0);
    return threshold;
  }

  void processFrame(const ComplexVector &in, ComplexMatrix out) {
    const auto &epsilon = std::numeric_limits<double>::epsilon;
    int h2 = (mHSize - 1) / 2;
    int v2 = (mVSize - 1) / 2;
    ArrayXcdMap frame(in.data(), mBins);
    ArrayXd mag = frame.abs().real();

    mV.block(0, 0, mBins, mHSize - 1) = mV.block(0, 1, mBins, mHSize - 1);
    mBuf.block(0, 0, mBins, mHSize - 1) = mBuf.block(0, 1, mBins, mHSize - 1);
    mHistory.block(0, 0, mBins, 2 * mHSize - 1) =
        mHistory.block(0, 1, mBins, 2 * mHSize - 1);
    ArrayXd padded =
        ArrayXd::Zero(mVSize + mVSize * int(ceil(mBins / double(mVSize))));
    ArrayXd resultV(padded.size());
    padded.segment(v2, mBins) = mag;
    mVMedianFilter.process(padded, resultV);
    mV.block(0, mHSize - 1, mBins, 1) = resultV.segment(v2, mBins);
    mBuf.block(0, mHSize - 1, mBins, 1) = frame;
    mHistory.block(0, mHSize + h2 - 1, mBins, 1) = mag;
    ArrayXd tmpRow = ArrayXd::Zero(2 * mHSize);
    for (int i = 0; i < mBins; i++) {
      mHMedianFilter.process(mHistory.row(i).transpose(), tmpRow);
      mH.row(i) = tmpRow.segment(h2, mHSize).transpose();
    }
    ArrayXXcd result(mBins, 3);

    ArrayXd harmonicMask = ArrayXd::Ones(mBins);
    ArrayXd percussiveMask = ArrayXd::Ones(mBins);
    ArrayXd residualMask = ArrayXd::Ones(mBins);

    switch (mMode) {
    case kClassic: {
      ArrayXd HV = mH.col(0) + mV.col(0);
      ArrayXd mult = (1.0 / HV.max(epsilon()));
      harmonicMask = (mH.col(0) * mult);
      percussiveMask = (mV.col(0) * mult);
      break;
    }
    case kCoupled: {
      harmonicMask = ((mH.col(0) / mV.col(0)) >
                      makeThreshold(mHThresholdX1, mHThresholdY1, mHThresholdX2,
                                    mHThresholdY2))
                         .cast<double>();
      percussiveMask = 1 - harmonicMask;
      break;
    }
    case kAdvanced: {
      harmonicMask = ((mH.col(0) / mV.col(0)) >
                      makeThreshold(mHThresholdX1, mHThresholdY1, mHThresholdX2,
                                    mHThresholdY2))
                         .cast<double>();
      percussiveMask = ((mV.col(0) / mH.col(0)) >
                        makeThreshold(mPThresholdX1, mPThresholdY1,
                                      mPThresholdX2, mPThresholdY2))
                           .cast<double>();
      residualMask = residualMask * (1 - harmonicMask);
      residualMask = residualMask * (1 - percussiveMask);
      ArrayXd maskNorm =
          (1. / (harmonicMask + percussiveMask + residualMask)).max(epsilon());
      harmonicMask = harmonicMask * maskNorm;
      percussiveMask = percussiveMask * maskNorm;
      residualMask = residualMask * maskNorm;
      break;
    }
    }

    result.col(0) = mBuf.col(0) * harmonicMask.min(1.0);
    result.col(1) = mBuf.col(0) * percussiveMask.min(1.0);
    result.col(2) = mBuf.col(0) * residualMask.min(1.0);
    out = ArrayXXcdToFluid(result)();
  }

  void setHThresholdX1(double x) {
    assert(0 <= x <= 1);
    mHThresholdX1 = x;
  }

  void setHThresholdX2(double x) {
    assert(0 <= x <= 1);
    mHThresholdX2 = x;
  }

  void setPThresholdX1(double x) {
    assert(0 <= x <= 1);
    mPThresholdX1 = x;
  }

  void setPThresholdX2(double x) {
    assert(0 <= x <= 1);
    mPThresholdX2 = x;
  }

  void setHThresholdY1(double y) { mHThresholdY1 = y; }

  void setHThresholdY2(double y) { mHThresholdY2 = y; }

  void setPThresholdY1(double y) { mPThresholdY1 = y; }

  void setPThresholdY2(double y) { mPThresholdY2 = y; }

private:
  size_t mBins;
  size_t mVSize;
  size_t mHSize;
  int mMode;
  MedianFilter mVMedianFilter;
  MedianFilter mHMedianFilter;
  ArrayXXd mH;
  ArrayXXd mV;
  ArrayXXd mHistory;
  ArrayXXcd mBuf;
  double mHThresholdX1;
  double mHThresholdY1;
  double mHThresholdX2;
  double mHThresholdY2;
  double mPThresholdX1;
  double mPThresholdY1;
  double mPThresholdX2;
  double mPThresholdY2;
};
} // namespace rthpss
} // namespace fluid
