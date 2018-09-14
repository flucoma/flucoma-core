#pragma once
#include "Novelty.hpp"
#include "data/FluidEigenMappings.hpp"
#include "data/FluidTensor.hpp"
#include <Eigen/Dense>

namespace fluid {
namespace novelty {

class NoveltySegmentation {
  using RealMatrix = FluidTensor<double, 2>;
  using RealVector = FluidTensor<double, 1>;

public:
  NoveltySegmentation(int kernelSize, double threshold)
      : mKernelSize(kernelSize), mThreshold(threshold) {
    assert(kernelSize % 2);
  }

  RealVector process(const RealMatrix &input) {
    using Eigen::ArrayXd;
    using Eigen::MatrixXd;
    using Eigen::VectorXd;
    using fluid::eigenmappings::FluidToMatrixXd;
    using std::vector;

    RealVector temp(input.extent(0));
    Novelty nov(mKernelSize);
    nov.process(input, temp);
    vector<double> peaks;
    for (int i = 1; i < temp.size() - 1; i++) {
      if (temp(i) > temp(i - 1) && temp(i) > temp(i + 1) &&
          temp(i) > mThreshold) {
        peaks.push_back(static_cast<double>(i));
      }
    }
    return RealVector(peaks.data(), peaks.size());
  }

private:
  int mKernelSize;
  double mThreshold;
};
} // namespace novelty
} // namespace fluid
