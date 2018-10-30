#pragma once
#include "Novelty.hpp"
#include "algorithms/ConvolutionTools.hpp"
#include "data/FluidEigenMappings.hpp"
#include "data/FluidTensor.hpp"
#include <Eigen/Dense>

namespace fluid {
namespace algorithm {

class NoveltySegmentation {
  using RealMatrix = FluidTensor<double, 2>;
  using RealVector = FluidTensor<double, 1>;

public:
  NoveltySegmentation(int kernelSize, double threshold, int filterSize)
      : mKernelSize(kernelSize), mThreshold(threshold),
        mFilterSize(filterSize) {
    assert(kernelSize % 2);
  }

  void process(const RealMatrix &input, RealVector &output) {
    using Eigen::ArrayXd;
    using Eigen::Map;
    using Eigen::MatrixXd;
    using Eigen::VectorXd;
    using algorithm::EdgeMode::kEdgeWrapCentre;
    using algorithm::convolveReal;
    using fluid::algorithm::FluidToMatrixXd;
    using std::vector;

    RealVector temp(input.extent(0));
    Novelty nov(mKernelSize);
    nov.process(input, temp);
    ArrayXd curve = Map<ArrayXd>(temp.data(), temp.size());
    if (mFilterSize > 0) {
      ArrayXd filter = ArrayXd::Constant(mFilterSize, 1.0 / mFilterSize);
      ArrayXd smoothed = ArrayXd::Zero(curve.size());
      convolveReal(smoothed.data(), curve.data(), curve.size(), filter.data(),
                   filter.size(), kEdgeWrapCentre);
      curve = smoothed;
    }
    curve /= curve.maxCoeff();

    for (int i = 1; i < curve.size() - 1; i++) {
      if (curve(i) > curve(i - 1) && curve(i) > curve(i + 1) &&
          curve(i) > mThreshold && i > mFilterSize / 2) {
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
