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

#include "../util/ConvolutionTools.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../util/Novelty.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>

namespace fluid {
namespace algorithm {

class NoveltySegmentation
{

public:
  using ArrayXd = Eigen::ArrayXd;

  NoveltySegmentation(index maxKernelSize, index maxFilterSize)
      : mFilterBufferStorage(maxFilterSize), mNovelty(maxKernelSize)
  {}

  void init(index kernelSize, double threshold, index filterSize, index nDims)
  {
    assert(kernelSize % 2);
    mThreshold = threshold;
    mFilterSize = filterSize;
    mNovelty.init(kernelSize, nDims);
    mFilterBuffer = mFilterBufferStorage.segment(0, mFilterSize);
    mFilterBuffer.setZero();
  }

  void setMinSliceLength(index val) { mDebounce = val; }

  double processFrame(const RealVectorView input)
  {
    double novelty = mNovelty.processFrame(_impl::asEigen<Eigen::Array>(input));
    double detected = 0.;
    if (mFilterSize > 1)
    {
      mFilterBuffer.segment(0, mFilterSize - 1) =
          mFilterBuffer.segment(1, mFilterSize - 1);
    }
    mPeakBuffer.segment(0, 2) = mPeakBuffer.segment(1, 2);
    mFilterBuffer(mFilterSize - 1) = novelty;
    mPeakBuffer(2) = mFilterBuffer.mean();
    if (mPeakBuffer(1) > mPeakBuffer(0) && mPeakBuffer(1) > mPeakBuffer(2) &&
        mPeakBuffer(1) > mThreshold && mDebounceCount == 0)
    {
      detected = 1.0;
      mDebounceCount = mDebounce;
    }
    else
    {
      if (mDebounceCount > 0) mDebounceCount--;
    }
    return detected;
  }

private:
  double  mThreshold{0.5};
  index   mFilterSize{3};
  ArrayXd mFilterBuffer;
  ArrayXd mFilterBufferStorage;
  ArrayXd mPeakBuffer{3};
  Novelty mNovelty;
  index   mDebounce{2};
  index   mDebounceCount{1};
};
} // namespace algorithm
} // namespace fluid
