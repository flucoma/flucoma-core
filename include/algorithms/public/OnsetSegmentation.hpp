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

#include "OnsetDetectionFunctions.hpp"
#include "WindowFuncs.hpp"
#include "../util/FFT.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../util/MedianFilter.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Eigen>
#include <algorithm>
#include <cassert>

namespace fluid {
namespace algorithm {

class OnsetSegmentation
{

public:
  using ArrayXd = Eigen::ArrayXd;
  using ArrayXcd = Eigen::ArrayXcd;

  OnsetSegmentation(index maxSize) : mODF{maxSize} {}

  void init(index windowSize, index fftSize, index filterSize)
  {
    mODF.init(windowSize, fftSize, filterSize);
    mDebounceCount = 1;
  }

  double processFrame(RealVectorView input, index function, index filterSize,
                      double threshold, index debounce = 0,
                      index frameDelta = 0)
  {
    double filteredFuncVal =
        mODF.processFrame(input, function, filterSize, frameDelta);
    double detected{0};

    if (filteredFuncVal > threshold && mPrevFuncVal < threshold &&
        mDebounceCount == 0)
    {
      detected = 1.0;
      mDebounceCount = debounce;
    }
    else
    {
      if (mDebounceCount > 0) mDebounceCount--;
    }
    mPrevFuncVal = filteredFuncVal;
    return detected;
  }

private:
  index                   mDebounceCount{1};
  OnsetDetectionFunctions mODF;
  double                  mPrevFuncVal{0.0};
};

} // namespace algorithm
} // namespace fluid
