/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright 2017-2019 University of Huddersfield.
Licensed under the BSD-3 License.
See LICENSE file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/

#pragma once

#include "OnsetDetectionFuncs.hpp"
#include "WindowFuncs.hpp"
#include "../util/ConvolutionTools.hpp"
#include "../util/FFT.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Eigen>
#include <algorithm>
#include <cassert>
#include <deque>

namespace fluid {
namespace algorithm {

class OnsetSegmentation
{

public:
  using ArrayXd = Eigen::ArrayXd;
  using ArrayXcd = Eigen::ArrayXcd;

  OnsetSegmentation(int maxSize)
      : mMaxSize(maxSize), mWindowStorage(maxSize), mFFT(maxSize),
        mFFTSize(maxSize), mWindowSize(maxSize), mHopSize(maxSize / 2)
  {
    makeWindow();
    initFilter();
  }

  void initFilter()
  {
    mFilter = std::deque<double>(mFilterSize, 0);
    mSorting = std::vector<int>(mFilterSize);
    std::iota(mSorting.begin(), mSorting.end(), 0);
  }

  void sortFilter()
  {
    for (int i = 1; i < mFilter.size(); ++i)
    {
      for (int j = i; j > 0 && mFilter[mSorting[j - 1]] > mFilter[mSorting[j]];
           --j)
      { std::swap(mSorting[j - 1], mSorting[j]); }
    }
  }

  void makeWindow()
  {
    mWindowStorage.setZero();
    WindowFuncs::map()[mWindowType](mWindowSize, mWindowStorage);
    mWindow = mWindowStorage.segment(0, mWindowSize);
    prevFrame = ArrayXcd::Zero(mFFTSize / 2 + 1);
    prevPrevFrame = ArrayXcd::Zero(mFFTSize / 2 + 1);
  }

  void updateParameters(int fftSize, int windowSize, int hopSize,
                        int frameDelta, int function, int filterSize,
                        double threshold, int debounce)
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
    if (mFilterSize != filterSize)
    {
      mFilterSize = filterSize;
      initFilter();
    }
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
    filteredFuncVal = funcVal - mFilter[mSorting[(mFilterSize - 1) / 2]];
    mFilter.push_back(funcVal);
    mFilter.pop_front();
    sortFilter();
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
  FFT                mFFT{1024};
  ArrayXd            mWindowStorage;
  ArrayXd            mWindow;
  int                mMaxSize{16384};
  int                mFFTSize{1024};
  int                mWindowSize{1024};
  int                mHopSize{512};
  int                mFrameDelta{0};
  int                mFunction{0};
  int                mFilterSize{5};
  double             mThreshold{0.1};
  int                mDebounce{2};
  int                mDebounceCount{1};
  std::deque<double> mFilter{5, 0};
  std::vector<int>   mSorting{5};
  ArrayXcd           prevFrame;
  ArrayXcd           prevPrevFrame;
  double             mPrevFuncVal{0.0};
  WindowTypes        mWindowType{WindowTypes::kHann};
};

}; // namespace algorithm
}; // namespace fluid
