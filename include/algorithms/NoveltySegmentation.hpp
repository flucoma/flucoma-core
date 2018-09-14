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

  RealVector process(const RealMatrix &input, RealVector &output) {
    using Eigen::ArrayXd;
    using Eigen::MatrixXd;
    using Eigen::VectorXd;
    using fluid::eigenmappings::FluidToMatrixXd;
    using std::vector;
    RealVector temp(input.extent(0));
    Novelty nov(mKernelSize);
    nov.process(input, temp);
    for (int i = 1; i < temp.size() - 1; i++) {
      if (temp(i) > temp(i - 1) && temp(i) > temp(i + 1) &&
          temp(i) > mThreshold) {
            output(i)  = 1;
      }
      else output(i)  = 0;
    }
  }

private:
  int mKernelSize;
  double mThreshold;
};
} // namespace novelty
} // namespace fluid
