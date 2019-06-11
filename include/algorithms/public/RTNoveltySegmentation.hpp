#pragma once

#include "../../data/TensorTypes.hpp"
#include "../util/ConvolutionTools.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../util/RTNovelty.hpp"
#include <Eigen/Dense>

namespace fluid {
namespace algorithm {

using _impl::asEigen;
using _impl::asFluid;
using Eigen::Array;

class RTNoveltySegmentation {

public:
  using ArrayXd = Eigen::ArrayXd;

  RTNoveltySegmentation(int maxKernelSize, int maxFilterSize)
      : mNovelty(maxKernelSize), mFilterBufferStorage(maxFilterSize),
        mPeakBuffer(3) {}

  void init(int kernelSize, double threshold, int filterSize, int nDims) {
    assert(kernelSize % 2);
    mThreshold = threshold;
    mFilterSize = filterSize;
    mNovelty.init(kernelSize, nDims);
    mFilterBuffer = mFilterBufferStorage.segment(0, mFilterSize);
    mFilterBuffer.setZero();
  }

  double processFrame(const RealVectorView input) {
    double novelty = mNovelty.processFrame(asEigen<Array>(input));
    if (mFilterSize > 1) {
      mFilterBuffer.segment(0, mFilterSize - 1) =
          mFilterBuffer.segment(1, mFilterSize - 1);
    }
    mPeakBuffer.segment(0, 2) = mPeakBuffer.segment(1, 2);
    mFilterBuffer(mFilterSize - 1) = novelty;
    mPeakBuffer(2) = mFilterBuffer.mean();
    if (mPeakBuffer(1) > mPeakBuffer(0) && mPeakBuffer(1) > mPeakBuffer(2) &&
        mPeakBuffer(1) > mThreshold)
      return 1;
    else
      return 0;
  }

private:
  double mThreshold;
  int mFilterSize;
  ArrayXd mFilterBuffer;
  ArrayXd mFilterBufferStorage;
  ArrayXd mPeakBuffer;
  RTNovelty mNovelty;
};
} // namespace algorithm
} // namespace fluid
