/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/

#pragma once

#include "AlgorithmUtils.hpp"
#include "FluidEigenMappings.hpp"
#include "../public/WindowFuncs.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/FluidMemory.hpp"
#include <Eigen/Core>

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

  Novelty(index maxSize, index maxDims, Allocator& alloc)
      : mKernel{maxSize, maxSize, alloc},
        mSimilarity{maxSize, maxSize, alloc}, mBufer{maxSize, maxDims, alloc}
  {}

  void init(index kernelSize, index nDims, Allocator& alloc)
  {
    assert(kernelSize % 2);
    mKernelSize = kernelSize;
    mNDims = nDims;
    createKernel(alloc);
    mSimilarity.setZero();
    mBufer.setZero();
    mInitialized = true;
  }

  template<typename EigenThing>
  double processFrame(const EigenThing& input, Allocator& alloc)
  {
    assert(mInitialized);
    mBufer.block(0, 0, mKernelSize - 1, mNDims) =
        mBufer.block(1, 0, mKernelSize - 1, mNDims);
    ScopedEigenMap<ArrayXXd> x(1, mNDims, alloc);
    x = mBufer.block(mKernelSize - 1, 0, 1, mNDims);
    ScopedEigenMap<VectorXd> in1(input.size(), alloc);
    in1 = input.matrix();
    mBufer.block(mKernelSize - 1, 0, 1, mNDims) = in1.transpose();
    ScopedEigenMap<VectorXd> tmp(mKernelSize, alloc);
    tmp.noalias() = mBufer.topLeftCorner(mKernelSize, mNDims) * input.matrix();
    ScopedEigenMap<VectorXd> norm(mKernelSize, alloc);
    norm.noalias() = mBufer.topLeftCorner(mKernelSize, mNDims)
                         .rowwise()
                         .norm()
                         .cwiseMax(epsilon) *
                     input.matrix().norm();
    norm = norm.cwiseMax(epsilon);
    tmp = (tmp.array() / norm.array()).matrix();
    mSimilarity.block(0, 0, mKernelSize - 1, mKernelSize - 1) =
        mSimilarity.block(1, 1, mKernelSize - 1, mKernelSize - 1);
    ScopedEigenMap<ArrayXXd> x1(mKernelSize, 1, alloc);
    x1 = mSimilarity.block(0, mKernelSize - 1, mKernelSize, 1);
    mSimilarity.block(0, mKernelSize - 1, mKernelSize, 1) = tmp;
    mSimilarity.block(mKernelSize - 1, 0, 1, mKernelSize) = tmp.transpose();
    double result =
        (mSimilarity.topLeftCorner(mKernelSize, mKernelSize).array() *
         mKernel.topLeftCorner(mKernelSize, mKernelSize))
            .sum();
    return result / mNorm;
  }

private:
  void createKernel(Allocator& alloc)
  {
    index                   h = (mKernelSize - 1) / 2;
    ScopedEigenMap<ArrayXd> gaussian(mKernelSize, alloc);
    WindowFuncs::map()[WindowFuncs::WindowTypes::kGaussian](mKernelSize,
                                                            gaussian);
    ScopedEigenMap<MatrixXd> tmp(mKernelSize, mKernelSize, alloc);
    tmp.noalias() = gaussian.matrix() * gaussian.matrix().transpose();
    tmp.block(h, 0, h + 1, h) *= -1;
    tmp.block(0, h, h, h + 1) *= -1;
    mKernel.topLeftCorner(mKernelSize, mKernelSize) = tmp.array();
    mNorm = mKernel.topLeftCorner(mKernelSize, mKernelSize).square().sum();
  }

  bool                     mInitialized{false};
  index                    mKernelSize{3};
  index                    mNDims{513};
  ScopedEigenMap<ArrayXXd> mKernel;
  ScopedEigenMap<MatrixXd> mSimilarity;
  ScopedEigenMap<MatrixXd> mBufer;
  double                   mNorm{1.};
};
} // namespace algorithm
} // namespace fluid
