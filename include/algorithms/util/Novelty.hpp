#pragma once

#include "../../data/FluidTensor.hpp"
#include "../public/Windows.hpp"
#include "Descriptors.hpp"
#include "FluidEigenMappings.hpp"
#include <Eigen/Dense>
#include <limits>

namespace fluid {
namespace algorithm {

  using Eigen::ArrayXd;
  using Eigen::ArrayXXd;
  using Eigen::MatrixXd;
  using Eigen::VectorXd;

// This implements Foote's novelty curve
class Novelty {
  using RealMatrixView = FluidTensor<double, 2>;
  using RealVectorView = FluidTensor<double, 1>;

public:
  Novelty(int kernelSize) : mKernelSize(kernelSize) {
    assert(kernelSize % 2);
    createKernel();
  }

  void process(const ArrayXXd &input, ArrayXd &output) {
    using std::vector;
    const auto &epsilon = std::numeric_limits<double>::epsilon;
    int nFrames = input.rows();
    int halfKernel = (mKernelSize - 1) / 2;
    MatrixXd featureMatrix = input.matrix();
    MatrixXd similarity =
        MatrixXd::Zero(nFrames + mKernelSize, nFrames + mKernelSize);
    MatrixXd tmp = featureMatrix * featureMatrix.transpose();
    VectorXd norm = featureMatrix.rowwise().norm().cwiseMax(epsilon());
    tmp = (tmp.array().rowwise() /= norm.transpose().array()).matrix();
    tmp = (tmp.array().colwise() /= norm.array()).matrix();
    similarity.block(halfKernel, halfKernel, nFrames, nFrames) = tmp;
    for (int i = 0; i < nFrames; i++) {
      output(i) =
          (similarity.block(i, i, mKernelSize, mKernelSize).array() * mKernel)
              .sum();
    }
    output = output / output.maxCoeff();
  }

private:
  int mKernelSize;
  Eigen::ArrayXXd mKernel;

  void createKernel() {
    int h = (mKernelSize - 1) / 2;
    ArrayXd gaussian = Eigen::Map<ArrayXd>(
        windowFuncs[WindowType::kGaussian](mKernelSize).data(), mKernelSize);
    MatrixXd tmp = gaussian.matrix() * gaussian.matrix().transpose();
    tmp.block(h, 0, h + 1, h) *= -1;
    tmp.block(0, h, h, h + 1) *= -1;
    mKernel = tmp.array();
  }
};
} // namespace algorithm
} // namespace fluid
