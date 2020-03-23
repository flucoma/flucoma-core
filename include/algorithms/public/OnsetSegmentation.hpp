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

#include "OnsetDetectionFuncs.hpp"
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

  OnsetSegmentation(index maxSize)
      : mFFT(maxSize), mWindowStorage(maxSize), mMaxSize(maxSize),
        mFFTSize(maxSize), mWindowSize(maxSize), mHopSize(maxSize / 2)
  {
    makeWindow();
    mFilter.init(mFilterSize);
  }

  void makeWindow()
  {
    mWindowStorage.setZero();
    WindowFuncs::map()[mWindowType](mWindowSize, mWindowStorage);
    mWindow = mWindowStorage.segment(0, mWindowSize);
    prevFrame = ArrayXcd::Zero(mFFTSize / 2 + 1);
    prevPrevFrame = ArrayXcd::Zero(mFFTSize / 2 + 1);
  }

  void updateParameters(index fftSize, index windowSize, index hopSize,
                        index frameDelta, index function, index filterSize,
                        double threshold, index debounce)
  {
    assert(fftSize <= mMaxSize);
    assert(windowSize <= mMaxSize);
    assert(windowSize <= fftSize);
    assert(hopSize <= mMaxSize);
    assert(frameDelta <= windowSize);
    assert(filterSize % 2);

    if (fftSize != mFFTSize)
    {
      mFFTSize = fftSize;
      mFFT.resize(mFFTSize);
      makeWindow();
    }
    if (windowSize != mWindowSize)
    {
      mWindowSize = windowSize;
      makeWindow();
    }

    mHopSize = hopSize;
    mFrameDelta = frameDelta;
    if (mFilter.size() != filterSize) mFilter.init(filterSize);
    mThreshold = threshold;
    mFunction = function;
    mDebounce = debounce;
  }

  double processFrame(RealVectorView input)
  {
    ArrayXd  in = _impl::asEigen<Eigen::Array>(input);
    double   funcVal = 0;
    double   filteredFuncVal = 0;
    double   detected = 0.;
    ArrayXcd frame = mFFT.process(in.segment(0, mWindowSize) * mWindow);
    auto     odf = static_cast<OnsetDetectionFuncs::ODF>(mFunction);
    if (mFunction > 1 && mFunction < 5 && mFrameDelta != 0)
    {
      ArrayXcd frame2 =
          mFFT.process(in.segment(mFrameDelta, mWindowSize) * mWindow);
      funcVal = OnsetDetectionFuncs::map()[odf](frame2, frame, frame);
    }
    else
    {
      funcVal =
          OnsetDetectionFuncs::map()[odf](frame, prevFrame, prevPrevFrame);
    }
    filteredFuncVal = funcVal - mFilter.processSample(funcVal);
    prevPrevFrame = prevFrame;
    prevFrame = frame;

    if (filteredFuncVal > mThreshold && mPrevFuncVal < mThreshold &&
        mDebounceCount == 0)
    {
      detected = 1.0;
      mDebounceCount = mDebounce;
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
  index        mMaxSize{16384};
  index        mFFTSize{1024};
  index        mWindowSize{1024};
  index        mHopSize{512};
  index        mFrameDelta{0};
  index        mFunction{0};
  index        mFilterSize{5};
  double       mThreshold{0.1};
  index        mDebounce{2};
  index        mDebounceCount{1};
  ArrayXcd     prevFrame;
  ArrayXcd     prevPrevFrame;
  double       mPrevFuncVal{0.0};
  WindowTypes  mWindowType{WindowTypes::kHann};
  MedianFilter mFilter;
};

} // namespace algorithm
} // namespace fluid
