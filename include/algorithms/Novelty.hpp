#pragma once
#include "Descriptors.hpp"
#include "Windows.hpp"
#include "data/FluidEigenMappings.hpp"
#include "data/FluidTensor.hpp"
#include <Eigen/Dense>
#include <limits>

namespace fluid {
namespace algorithm {

// This implements Foote's novelty curve
class Novelty {
  using RealMatrix = FluidTensor<double, 2>;
  using RealVector = FluidTensor<double, 1>;

public:
  Novelty(int kernelSize) : mKernelSize(kernelSize) {
    assert(kernelSize % 2);
    createKernel();
  }

  void process(const RealMatrix &input, RealVector &output) {
    using Eigen::ArrayXd;
    using Eigen::MatrixXd;
    using Eigen::VectorXd;
    using ArrayXdMap =
        Eigen::Map<Eigen::Array<double, Eigen::Dynamic, Eigen::RowMajor>>;
    using fluid::algorithm::Descriptors;
    using fluid::algorithm::FluidToMatrixXd;
    using std::vector;

    const auto &epsilon = std::numeric_limits<double>::epsilon;
    int nFrames = input.extent(0);
    int halfKernel = (mKernelSize - 1) / 2;
    MatrixXd featureMatrix = FluidToMatrixXd(input)();
    MatrixXd similarity =
        MatrixXd::Zero(nFrames + mKernelSize, nFrames + mKernelSize);
    MatrixXd tmp = featureMatrix * featureMatrix.transpose();
    VectorXd norm = featureMatrix.rowwise().norm().cwiseMax(epsilon());
    tmp = (tmp.array().rowwise() /= norm.transpose().array()).matrix();
    tmp = (tmp.array().colwise() /= norm.array()).matrix();
    similarity.block(halfKernel, halfKernel, nFrames, nFrames) = tmp;
    ArrayXd novelty(nFrames);
    for (int i = 0; i < nFrames; i++) {
      novelty(i) =
          (similarity.block(i, i, mKernelSize, mKernelSize).array() * mKernel)
              .sum();
    }
    novelty = novelty / novelty.maxCoeff();
    ArrayXdMap(output.data(), nFrames) = novelty;
  }

private:
  int mKernelSize;
  Eigen::ArrayXXd mKernel;

  void createKernel() {
    using Eigen::ArrayXd;
    using Eigen::Map;
    using Eigen::MatrixXd;
    using algorithm::WindowType;
    using algorithm::windowFuncs;
    int h = (mKernelSize - 1) / 2;
    ArrayXd gaussian = Map<ArrayXd>(
        windowFuncs[WindowType::kGaussian](mKernelSize).data(), mKernelSize);
    MatrixXd tmp = gaussian.matrix() * gaussian.matrix().transpose();
    tmp.block(h, 0, h + 1, h) *= -1;
    tmp.block(0, h, h, h + 1) *= -1;
    mKernel = tmp.array();
  }
};
} // namespace algorithm
} // namespace fluid
