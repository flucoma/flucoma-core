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

#include "NoveltyCurve.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>

namespace fluid {
namespace algorithm {

class NoveltySegmentation
{

public:
  using ArrayXd = Eigen::ArrayXd;

  NoveltySegmentation(index maxKernelSize, index maxFilterSize)
      : mNovelty(maxFilterSize, maxKernelSize)
  {}

  void init(index kernelSize, index filterSize, index nDims)
  {
    mNovelty.init(kernelSize, filterSize, nDims);
    mDebounceCount = 1;
  }

  double processFrame(const RealVectorView input, double threshold,
                      index minSliceLength)
  {
    double detected = 0.;

    mPeakBuffer.segment(0, 2) = mPeakBuffer.segment(1, 2);
    mPeakBuffer(2) = mNovelty.processFrame(input);

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
  NoveltyCurve mNovelty;
  ArrayXd      mPeakBuffer{3};
  index        mDebounceCount{1};
};
} // namespace algorithm
} // namespace fluid
