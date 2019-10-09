#pragma once

#include "../util/AlgorithmUtils.hpp"
#include "../../data/FluidTensor.hpp"
#include "../public/WindowFuncs.hpp"
#include "FluidEigenMappings.hpp"

#include <Eigen/Core>
#include <limits>

namespace fluid {
namespace algorithm {

// This implements Foote's novelty curve
class Novelty {

public:
  using ArrayXd = Eigen::ArrayXd;
  using ArrayXXd = Eigen::ArrayXXd;
  using MatrixXd = Eigen::MatrixXd;
  using VectorXd = Eigen::VectorXd;

  Novelty(int maxSize) : mKernelStorage(maxSize, maxSize) {}

  void init(int kernelSize, int nDims) {
    assert(kernelSize % 2);
    mKernelSize = kernelSize;
    mNDims = nDims;
    createKernel();
    mSimilarity = MatrixXd::Zero(mKernelSize, mKernelSize);
    mBufer = MatrixXd::Zero(mKernelSize, nDims);
  }

  double processFrame(const ArrayXd &input) {
    using std::vector;
    int halfKernel = (mKernelSize - 1) / 2;
    mBufer.block(0, 0, mKernelSize - 1, mNDims) =
        mBufer.block(1, 0, mKernelSize - 1, mNDims);

    ArrayXXd x = mBufer.block(mKernelSize - 1, 0, 1, mNDims);
    VectorXd in1 = input.matrix();
    mBufer.block(mKernelSize - 1, 0, 1, mNDims) = in1.transpose();
    VectorXd tmp = mBufer * input.matrix();

    VectorXd norm = mBufer.rowwise().norm().cwiseMax(epsilon) * input.matrix().norm();
    norm = norm.cwiseMax(epsilon);
    tmp = (tmp.array() / norm.array()).matrix();
    mSimilarity.block(0, 0, mKernelSize - 1, mKernelSize - 1) =
        mSimilarity.block(1, 1, mKernelSize - 1, mKernelSize - 1);
    ArrayXXd x1 = mSimilarity.block(0, mKernelSize -1, mKernelSize, 1);
    mSimilarity.block(0, mKernelSize -1, mKernelSize, 1) = tmp;
    mSimilarity.block(mKernelSize - 1, 0, 1, mKernelSize) = tmp.transpose();
    double result = (mSimilarity.array() * mKernel).sum();
    return result / mNorm;
  }

private:
  int mKernelSize;
  int mNDims;
  ArrayXXd mKernel;
  ArrayXXd mKernelStorage;
  MatrixXd mSimilarity;
  MatrixXd mBufer;
  int mNorm;

  void createKernel() {
    mKernel = mKernelStorage.block(0, 0, mKernelSize, mKernelSize);
    int h = (mKernelSize - 1) / 2;
    ArrayXd gaussian = ArrayXd::Zero(mKernelSize);
    WindowFuncs::map()[WindowFuncs::WindowTypes::kGaussian](mKernelSize, gaussian);
    MatrixXd tmp = gaussian.matrix() * gaussian.matrix().transpose();
    tmp.block(h, 0, h + 1, h) *= -1;
    tmp.block(0, h, h, h + 1) *= -1;
    mKernel = tmp.array();
    mNorm = mKernel.square().sum();
  }

};
} // namespace algorithm
} // namespace fluid
