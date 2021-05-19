/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright 2017-2019 University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/

#pragma once

#include "../util/AlgorithmUtils.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../util/MedianFilter.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>

namespace fluid {
namespace algorithm {

class HPSS
{
public:
  using ArrayXXd = Eigen::ArrayXXd;
  using ArrayXXcd = Eigen::ArrayXXcd;
  using ArrayXcd = Eigen::ArrayXcd;

  enum HPSSMode { kClassic, kCoupled, kAdvanced };

  HPSS(index maxFFTSize, index maxHSize)
      : mMaxH(maxFFTSize / 2 + 1, maxHSize),
        mMaxV(maxFFTSize / 2 + 1, maxHSize),
        mMaxBuf(maxFFTSize / 2 + 1, maxHSize)
  {
    mMaxH.setZero();
    mMaxV.setZero();
    mMaxBuf.setZero();
  }

  void init(index nBins, index hSize)
  {
    using namespace Eigen;
    assert(hSize % 2);
    assert(nBins <= mMaxBuf.rows());
    assert(hSize <= mMaxBuf.cols());

    mH = mMaxH.block(0, 0, nBins, hSize);
    mV = mMaxV.block(0, 0, nBins, hSize);
    mBuf = mMaxBuf.block(0, 0, nBins, hSize);
    mH.setZero();
    mV.setZero();
    mBuf.setZero();

    mHFilters = std::vector<MedianFilter>(asUnsigned(nBins));
    for (index i = 0; i < nBins; i++) { mHFilters[asUnsigned(i)].init(hSize); }
    mInitialized = true;
  }

  void processFrame(const ComplexVectorView in, ComplexMatrixView out,
                    index vSize, index hSize, index mode, double hThresholdX1,
                    double hThresholdY1, double hThresholdX2,
                    double hThresholdY2, double pThresholdX1,
                    double pThresholdY1, double pThresholdX2,
                    double pThresholdY2)
  {
    using namespace Eigen;
    assert(mInitialized);

    index    h2 = (hSize - 1) / 2;
    index    v2 = (vSize - 1) / 2;
    index    nBins = in.size();
    ArrayXcd frame = _impl::asEigen<Array>(in);
    ArrayXd  mag = frame.abs().real();

    mV.block(0, 0, nBins, hSize - 1) = mV.block(0, 1, nBins, hSize - 1);
    mH.block(0, 0, nBins, hSize - 1) = mH.block(0, 1, nBins, hSize - 1);
    mBuf.block(0, 0, nBins, hSize - 1) = mBuf.block(0, 1, nBins, hSize - 1);

    ArrayXd padded = ArrayXd::Zero(2 * vSize + nBins);
    ArrayXd resultV = ArrayXd::Zero(padded.size());
    ArrayXd tmp = ArrayXd::Zero(padded.size());

    padded.segment(v2, nBins) = mag;
    mVFilter.init(vSize);
    for (index i = 0; i < padded.size(); i++)
    { tmp(i) = mVFilter.processSample(padded(i)); }
    mV.block(0, hSize - 1, nBins, 1) = tmp.segment(v2 * 3, nBins);
    mBuf.block(0, hSize - 1, nBins, 1) = frame;
    ArrayXd tmpRow = ArrayXd::Zero(2 * hSize);
    for (index i = 0; i < nBins; i++)
    { mH(i, h2 + 1) = mHFilters[asUnsigned(i)].processSample(mag(i)); }
    ArrayXXcd result(nBins, 3);
    ArrayXd   harmonicMask = ArrayXd::Ones(nBins);
    ArrayXd   percussiveMask = ArrayXd::Ones(nBins);
    ArrayXd   residualMask =
        mode == kAdvanced ? ArrayXd::Ones(nBins) : ArrayXd::Zero(nBins);
    switch (mode)
    {
    case kClassic: {
      ArrayXd HV = mH.col(0) + mV.col(0);
      ArrayXd mult = (1.0 / HV.max(epsilon));
      harmonicMask = (mH.col(0) * mult);
      percussiveMask = (mV.col(0) * mult);
      break;
    }
    case kCoupled: {
      harmonicMask = ((mH.col(0) / mV.col(0)) >
                      makeThreshold(nBins, hThresholdX1, hThresholdY1,
                                    hThresholdX2, hThresholdY2))
                         .cast<double>();
      percussiveMask = 1 - harmonicMask;
      break;
    }
    case kAdvanced: {
      harmonicMask = ((mH.col(0) / mV.col(0)) >
                      makeThreshold(nBins, hThresholdX1, hThresholdY1,
                                    hThresholdX2, hThresholdY2))
                         .cast<double>();
      percussiveMask = ((mV.col(0) / mH.col(0)) >
                        makeThreshold(nBins, pThresholdX1, pThresholdY1,
                                      pThresholdX2, pThresholdY2))
                           .cast<double>();
      residualMask = residualMask * (1 - harmonicMask);
      residualMask = residualMask * (1 - percussiveMask);
      ArrayXd maskNorm =
          (1. / (harmonicMask + percussiveMask + residualMask)).max(epsilon);
      harmonicMask = harmonicMask * maskNorm;
      percussiveMask = percussiveMask * maskNorm;
      residualMask = residualMask * maskNorm;
      break;
    }
    }
    result.col(0) = mBuf.col(0) * harmonicMask.min(1.0);
    result.col(1) = mBuf.col(0) * percussiveMask.min(1.0);
    result.col(2) = mBuf.col(0) * residualMask.min(1.0);
    out = _impl::asFluid(result);
  }
  bool initialized() { return mInitialized; }

private:
  Eigen::ArrayXd makeThreshold(index nBins, double x1, double y1, double x2,
                               double y2)
  {
    using namespace Eigen;
    ArrayXd threshold = ArrayXd::Ones(nBins);
    index   kneeStart = static_cast<index>(std::floor(x1 * nBins));
    index   kneeEnd = static_cast<index>(std::floor(x2 * nBins));
    index   kneeLength = kneeEnd - kneeStart;
    threshold.segment(0, kneeStart) =
        ArrayXd::Constant(kneeStart, 10).pow(y1 / 20.0);
    threshold.segment(kneeStart, kneeLength) =
        ArrayXd::Constant(kneeLength, 10)
            .pow(ArrayXd::LinSpaced(kneeLength, y1, y2) / 20.0);
    threshold.segment(kneeEnd, nBins - kneeEnd) =
        ArrayXd::Constant(nBins - kneeEnd, 10).pow(y2 / 20.0);
    return threshold;
  }

  std::vector<MedianFilter> mHFilters;
  MedianFilter              mVFilter;

  ArrayXXd  mMaxH;
  ArrayXXd  mMaxV;
  ArrayXXcd mMaxBuf;
  ArrayXXd  mV;
  ArrayXXd  mH;
  ArrayXXcd mBuf;
  bool      mInitialized{false};
};
} // namespace algorithm
} // namespace fluid
