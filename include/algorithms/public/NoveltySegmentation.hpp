#pragma once

#include "../../data/TensorTypes.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../util/ConvolutionTools.hpp"
#include "../util/Novelty.hpp"
#include <Eigen/Dense>

namespace fluid {
namespace algorithm {

using _impl::asEigen;
using _impl::asFluid;

class NoveltySegmentation {

public:
  NoveltySegmentation(int kernelSize, double threshold, int filterSize)
      : mKernelSize(kernelSize), mThreshold(threshold),
        mFilterSize(filterSize) {
    assert(kernelSize % 2);
  }

  void process(const RealMatrixView input, RealVectorView output) {
    using Eigen::ArrayXd;
    using Eigen::Array;

    ArrayXd curve(input.extent(0));
    Novelty nov(mKernelSize);
    nov.process(asEigen<Array>(input), curve);
    if (mFilterSize > 0) {
      ArrayXd filter = ArrayXd::Constant(mFilterSize, 1.0 / mFilterSize);
      ArrayXd smoothed = ArrayXd::Zero(curve.size());
      convolveReal(smoothed.data(), curve.data(), curve.size(), filter.data(),
                   filter.size(), EdgeMode::kEdgeWrapCentre);
      curve = smoothed;
    }
    curve /= curve.maxCoeff();
    for (int i = mFilterSize / 2; i < curve.size() - 1; i++) {
      if (curve(i) > curve(i - 1) && curve(i) > curve(i + 1) &&
          curve(i) > mThreshold) {
        output(i - mFilterSize / 2) = 1;
      } else
        output(i - mFilterSize / 2) = 0;
    }
  }

private:
  int mKernelSize;
  double mThreshold;
  int mFilterSize;
};
} // namespace algorithm
} // namespace fluid
