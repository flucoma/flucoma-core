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

#include "FluidEigenMappings.hpp"
#include "../public/WindowFuncs.hpp"
#include "../util/AlgorithmUtils.hpp"
#include "../../data/FluidTensor.hpp"
#include <Eigen/Core>
#include <limits>

namespace fluid {
namespace algorithm {

// This implements Foote's novelty curve
class Novelty
{

public:
  using ArrayXd = Eigen::ArrayXd;
  using ArrayXXd = Eigen::ArrayXXd;
  using MatrixXd = Eigen::MatrixXd;
  using VectorXd = Eigen::VectorXd;

  Novelty(index maxSize) : mKernelStorage(maxSize, maxSize) {}

  void init(index kernelSize, index nDims)
  {
    assert(kernelSize % 2);
    mKernelSize = kernelSize;
    mNDims = nDims;
    createKernel();
    mSimilarity = MatrixXd::Zero(mKernelSize, mKernelSize);
    mBufer = MatrixXd::Zero(mKernelSize, nDims);
  }

  double processFrame(const ArrayXd& input)
  {
    using std::vector;
    mBufer.block(0, 0, mKernelSize - 1, mNDims) =
        mBufer.block(1, 0, mKernelSize - 1, mNDims);
    ArrayXXd x = mBufer.block(mKernelSize - 1, 0, 1, mNDims);
    VectorXd in1 = input.matrix();
    mBufer.block(mKernelSize - 1, 0, 1, mNDims) = in1.transpose();
    VectorXd tmp = mBufer * input.matrix();
    VectorXd norm =
        mBufer.rowwise().norm().cwiseMax(epsilon) * input.matrix().norm();
    norm = norm.cwiseMax(epsilon);
    tmp = (tmp.array() / norm.array()).matrix();
    mSimilarity.block(0, 0, mKernelSize - 1, mKernelSize - 1) =
        mSimilarity.block(1, 1, mKernelSize - 1, mKernelSize - 1);
    ArrayXXd x1 = mSimilarity.block(0, mKernelSize - 1, mKernelSize, 1);
    mSimilarity.block(0, mKernelSize - 1, mKernelSize, 1) = tmp;
    mSimilarity.block(mKernelSize - 1, 0, 1, mKernelSize) = tmp.transpose();
    double result = (mSimilarity.array() * mKernel).sum();
    return result / mNorm;
  }

private:
  void createKernel()
  {
    mKernel = mKernelStorage.block(0, 0, mKernelSize, mKernelSize);
    index   h = (mKernelSize - 1) / 2;
    ArrayXd gaussian = ArrayXd::Zero(mKernelSize);
    WindowFuncs::map()[WindowFuncs::WindowTypes::kGaussian](mKernelSize,
                                                            gaussian);
    MatrixXd tmp = gaussian.matrix() * gaussian.matrix().transpose();
    tmp.block(h, 0, h + 1, h) *= -1;
    tmp.block(0, h, h, h + 1) *= -1;
    mKernel = tmp.array();
    mNorm = mKernel.square().sum();
  }

  index    mKernelSize{3};
  index    mNDims{513};
  ArrayXXd mKernel;
  ArrayXXd mKernelStorage;
  MatrixXd mSimilarity;
  MatrixXd mBufer;
  double   mNorm{1.};
};
} // namespace algorithm
} // namespace fluid
