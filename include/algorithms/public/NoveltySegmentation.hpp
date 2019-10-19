/*
Copyright 2017-2019 University of Huddersfield.
Licensed under the BSD-3 License.
See LICENSE file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/

#pragma once

#include "../../data/TensorTypes.hpp"
#include "../util/ConvolutionTools.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../util/Novelty.hpp"

#include <Eigen/Core>

namespace fluid {
namespace algorithm {

class NoveltySegmentation
{

public:
  using ArrayXd = Eigen::ArrayXd;

  NoveltySegmentation(int maxKernelSize, int maxFilterSize)
      : mNovelty(maxKernelSize), mFilterBufferStorage(maxFilterSize)
  {}

  void init(int kernelSize, double threshold, int filterSize, int nDims)
  {
    assert(kernelSize % 2);
    mThreshold = threshold;
    mFilterSize = filterSize;
    mNovelty.init(kernelSize, nDims);
    mFilterBuffer = mFilterBufferStorage.segment(0, mFilterSize);
    mFilterBuffer.setZero();
  }

  double processFrame(const RealVectorView input)
  {
    double novelty = mNovelty.processFrame(_impl::asEigen<Eigen::Array>(input));
    if (mFilterSize > 1)
    {
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
  double  mThreshold{0.5};
  int     mFilterSize{3};
  ArrayXd mFilterBuffer;
  ArrayXd mFilterBufferStorage;
  ArrayXd mPeakBuffer{3};
  Novelty mNovelty;
};
} // namespace algorithm
} // namespace fluid
