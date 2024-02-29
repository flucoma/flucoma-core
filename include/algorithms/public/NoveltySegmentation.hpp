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

#include "NoveltyFeature.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/FluidMemory.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>

namespace fluid {
namespace algorithm {

class NoveltySegmentation
{

public:
  using ArrayXd = Eigen::ArrayXd;

  NoveltySegmentation(index maxKernelSize, index maxDims, index maxFilterSize,
      Allocator& alloc = FluidDefaultAllocator())
      : mNovelty(maxKernelSize, maxDims, maxFilterSize, alloc),
        mPeakBuffer(3, alloc)
  {}

  void init(index kernelSize, index filterSize, index nDims,
      Allocator& alloc = FluidDefaultAllocator())
  {
    mNovelty.init(kernelSize, filterSize, nDims, alloc);
    mDebounceCount = 0;
    mPeakBuffer.setZero();
  }

  double processFrame(const RealVectorView input, double threshold,
      index minSliceLength, Allocator& alloc = FluidDefaultAllocator())
  {
    double detected = 0.;

    mPeakBuffer.segment(0, 2) = mPeakBuffer.segment(1, 2);
    mPeakBuffer(2) = mNovelty.processFrame(input, alloc);

    if (mPeakBuffer(1) > mPeakBuffer(0) && mPeakBuffer(1) > mPeakBuffer(2) &&
        mPeakBuffer(1) > threshold && mDebounceCount == 0)
    {
      detected = 1.0;
      mDebounceCount = minSliceLength;
    }
    else
    {
      if (mDebounceCount > 0) mDebounceCount--;
    }
    return detected;
  }

private:
  NoveltyFeature          mNovelty;
  ScopedEigenMap<ArrayXd> mPeakBuffer;
  index                   mDebounceCount{0};
};
} // namespace algorithm
} // namespace fluid
