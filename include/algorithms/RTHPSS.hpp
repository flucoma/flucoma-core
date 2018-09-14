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
  RTHPSS(int nBins, int vSize, int hSize, double hThreshold, double pThreshold)
      : mVSize(vSize), mHSize(hSize), mBins(nBins), mVMedianFilter(vSize),
        mHMedianFilter(hSize), mHThreshold(hThreshold),
        mPThreshold(pThreshold) {
    assert(mVSize % 2);
    assert(mHSize % 2);
    mH = ArrayXXd::Zero(mBins, hSize);
    mV = ArrayXXd::Zero(mBins, hSize);
    mBuf = ArrayXXcd::Zero(mBins, hSize);
    mHistory = ArrayXXd::Zero(mBins, 2 * hSize);
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
    ArrayXXcd result(mBins, 2);
    ArrayXd HV = mH.col(0) + mV.col(0);
    ArrayXd mult = 1.0 / HV.max(epsilon());
    if (mHThreshold == 0) {
      result.col(0) = mBuf.col(0) * (mH.col(0) * mult).min(1.0);
    } else {
      ArrayXd bMask = ((mH.col(0) / mV.col(0)) > mHThreshold).cast<double>();
      result.col(0) = mBuf.col(0) * bMask;
    }

    if (mPThreshold == 0) {
      result.col(1) = mBuf.col(0) * (mV.col(0) * mult).min(1.0);
    } else {
      ArrayXd bMask = ((mV.col(0) / mH.col(0)) > mPThreshold).cast<double>();
      result.col(1) = mBuf.col(0) * bMask;
    }
    out = ArrayXXcdToFluid(result)();
  }

  void setHThreshold(double threshold) {
    assert(0 <= threshold <= 100);
    mHThreshold = threshold;
  }

  void setPThreshold(double threshold) {
    assert(0 <= threshold <= 100);
    mPThreshold = threshold;
  }

private:
  size_t mVSize;
  size_t mHSize;
  size_t mBins;
  MedianFilter mVMedianFilter;
  MedianFilter mHMedianFilter;
  ArrayXXd mH;
  ArrayXXd mV;
  ArrayXXd mHistory;
  ArrayXXcd mBuf;
  double mHThreshold;
  double mPThreshold;
};
} // namespace rthpss
} // namespace fluid
