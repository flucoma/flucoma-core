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

#include "WindowFuncs.hpp"
#include "../util/FFT.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../util/MedianFilter.hpp"
#include "../util/OnsetDetectionFuncs.hpp"
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

  OnsetSegmentation(index maxSize) : mFFT(maxSize), mWindowStorage(maxSize) {}

  void init(index windowSize, index fftSize)
  {
    makeWindow(windowSize);
    prevFrame = ArrayXcd::Zero(fftSize / 2 + 1);
    prevPrevFrame = ArrayXcd::Zero(fftSize / 2 + 1);
    mFFT.resize(fftSize);
    mDebounceCount = 1;
    mPrevFuncVal = 0;
    mInitialized = true;
  }

  void makeWindow(index windowSize)
  {
    mWindowStorage.setZero();
    WindowFuncs::map()[mWindowType](windowSize, mWindowStorage);
    mWindow = mWindowStorage.segment(0, windowSize);
    mWindowSize = windowSize;
  }

  double processFrame(RealVectorView input, index function, index filterSize,
                      double threshold, index debounce = 0,
                      index frameDelta = 0)
  {
    assert(mInitialized);
    ArrayXd in = _impl::asEigen<Eigen::Array>(input);
    double  funcVal = 0;
    double  filteredFuncVal = 0;
    double  detected = 0.;
    if (filterSize >= 3 &&
        (!mFilter.initialized() || filterSize != mFilter.size()))
      mFilter.init(filterSize);

    ArrayXcd frame = mFFT.process(in.segment(0, mWindowSize) * mWindow);
    auto     odf = static_cast<OnsetDetectionFuncs::ODF>(function);
    if (function > 1 && function < 5 && frameDelta != 0)
    {
      ArrayXcd frame2 =
          mFFT.process(in.segment(frameDelta, mWindowSize) * mWindow);
      funcVal = OnsetDetectionFuncs::map()[odf](frame2, frame, frame);
    }
    else
    {
      funcVal =
          OnsetDetectionFuncs::map()[odf](frame, prevFrame, prevPrevFrame);
    }
    if (filterSize >= 3)
      filteredFuncVal = funcVal - mFilter.processSample(funcVal);
    else
      filteredFuncVal = funcVal - mPrevFuncVal;

    prevPrevFrame = prevFrame;
    prevFrame = frame;

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
  using WindowTypes = WindowFuncs::WindowTypes;
  FFT          mFFT{1024};
  ArrayXd      mWindowStorage;
  ArrayXd      mWindow;
  index        mWindowSize{1024};
  index        mDebounceCount{1};
  ArrayXcd     prevFrame;
  ArrayXcd     prevPrevFrame;
  double       mPrevFuncVal{0.0};
  WindowTypes  mWindowType{WindowTypes::kHann};
  MedianFilter mFilter;
  bool         mInitialized{false};
};

} // namespace algorithm
} // namespace fluid
